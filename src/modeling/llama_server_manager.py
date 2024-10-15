import os
import platform
import subprocess
import threading
import asyncio
import httpx
import psutil
import shutil

# Environment Variables
LLAMA_PORT = int(os.getenv("LLAMA_PORT", 8001))
GENERAL_MODEL_PATH = os.getenv("general", "efs/models/Llama-3.1.gguf")
LLAMA_CPP_HOME = os.getenv("LLAMA_CPP_HOME", "/opt/cx_intelligence/aiaas/compiled_llama_cpp")
LLAMA_CPP_PATH = os.path.join(LLAMA_CPP_HOME, "bin/llama-server")
LLAMA_SOURCE_FOLDER = os.getenv("LLAMA_SOURCE_FOLDER", "efs/frameworks/llama.cpp")
HOST = os.getenv("HOST", "0.0.0.0")
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "99"))
NUMBER_OF_CORES = os.cpu_count()
TOTAL_BATCH_SIZE = int(os.getenv("TOTAL_BATCH_SIZE", "8196"))
TOTAL_UBATCH_SIZE = int(os.getenv("TOTAL_UBATCH_SIZE", "2048"))
TOTAL_THREADS_BATCH = int(os.getenv("TOTAL_THREADS_BATCH", "4"))
MAX_CALLS = int(os.getenv("MAX_CALLS", "35"))
WORKERS = "-j4"  # Setting it to use 4 workers for compilation
LLAMA_REPO_URL = "https://github.com/ggerganov/llama.cpp.git"  # The URL to the llama.cpp repository


def check_possible_paths():
    """
    Check if the llama-server binary exists in possible paths.
    """
    if os.path.exists(LLAMA_CPP_PATH):
        return LLAMA_CPP_PATH
    return ""


def remove_directory(directory):
    """
    Remove a directory if it exists.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)


def is_cuda_available():
    """
    Check if CUDA is available on the system by running nvcc --version.
    """
    try:
        result = subprocess.run(["nvcc", "--version"], check=True, capture_output=True)
        return "Cuda compilation tools" in result.stdout.decode()
    except FileNotFoundError:
        return False


def clone_llama_cpp_repo():
    """
    Clones the llama.cpp repository into the LLAMA_SOURCE_FOLDER.
    """
    if not os.path.exists(LLAMA_SOURCE_FOLDER):
        print(f"Cloning llama.cpp into {LLAMA_SOURCE_FOLDER}...")
        os.makedirs(LLAMA_SOURCE_FOLDER, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", LLAMA_REPO_URL, LLAMA_SOURCE_FOLDER],
                check=True,
                capture_output=True,
            )
            print(f"Successfully cloned the llama.cpp repo to {LLAMA_SOURCE_FOLDER}.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning the llama.cpp repository: {e}")
            raise e


def compile_llama_cpp():
    """
    Compiles the llama-server using CMake and GPU acceleration (if available).
    """
    global LLAMA_CPP_PATH
    LLAMA_CPP_PATH = check_possible_paths()

    if LLAMA_CPP_PATH == "":
        # Clone the repo if necessary
        clone_llama_cpp_repo()

        # Remove any existing directory and create a clean one
        remove_directory(LLAMA_CPP_HOME)
        os.makedirs(LLAMA_CPP_HOME, exist_ok=True)
        os.chdir(LLAMA_CPP_HOME)

        # Determine the correct GPU flag based on the system
        gpu_flag = ""
        if is_cuda_available():
            gpu_flag = "-DGGML_CUDA=ON"
        elif platform.system() == "Darwin":
            gpu_flag = "-DGGML_METAL=ON"

        try:
            # Configure CMake with the correct GPU support
            print("Configuring CMake...")
            source_command = subprocess.run(
                ["cmake", "-B", ".", "-S", LLAMA_SOURCE_FOLDER, gpu_flag],
                check=True,
                capture_output=True,
            )
            print(source_command.stdout.decode())
            print(source_command.stderr.decode())

            # Build llama-server target
            print("Building llama-server...")
            build_command = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "--target", "llama-server", WORKERS],
                check=True,
                capture_output=True,
            )
            print(build_command.stdout.decode())
            print(build_command.stderr.decode())

            # After compiling, check if the binary exists
            LLAMA_CPP_PATH = check_possible_paths()
            if LLAMA_CPP_PATH == "":
                raise RuntimeError("Could not find compiled llama-server binary.")

        except subprocess.CalledProcessError as e:
            print(f"Error during CMake or build process: {e}")
            exit(1)


def kill_process_on_port(port):
    """
    Kill the process running on the specified port.
    """
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            pid = conn.pid
            if pid:
                try:
                    os.kill(pid, 9)
                    print(f"Killed process {pid} on port {port}.")
                except Exception as e:
                    print(f"Failed to kill process {pid} on port {port}: {e}")
            return
    print(f"No process found running on port {port}.")


def copy_llama_binary(server_number):
    """
    Copy the llama-server binary to the specified path.
    """

    destination_path = f"efs/bin/llama-{server_number}/llama-server"
    if not os.path.exists(destination_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(LLAMA_CPP_PATH, destination_path)
        print(f"Copied llama-server binary to {destination_path}.")


def is_server_up(port):
    """
    Check if the server is up by sending a simple request.
    """
    url = f"http://localhost:{port}/health"
    try:
        response = httpx.get(url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


class LlamaServerManager:
    """
    Manages the llama-server instances, including starting, stopping, and switching between them.
    """

    def __init__(self):
        # Always spin up 1 additional server for standby
        self.number_of_requested_servers = int(os.getenv('NUMBER_OF_SERVERS', "1"))
        self.number_of_servers = self.number_of_requested_servers + 1  # Always 1 extra for standby
        self.servers = []
        self.server_calls = []
        self.server_commands = []  # To keep track of commands for each server
        self.current_server_index = 0
        self.active_requests = 0
        self.lock = threading.Lock()
        self.switch_in_progress = False
        self.is_request_in_progress = [False] * self.number_of_servers

        # Initialize ports and active server indices
        self.ports = [LLAMA_PORT + i for i in range(self.number_of_servers)]
        self.active_server_indices = list(range(self.number_of_servers))  # All servers are initially active

    async def start_server(self, path, port):
        """
        Start a llama-server instance with the specified resource configuration.
        """
        gpu_layers_per_server = str(max(1, GPU_LAYERS // self.number_of_servers))
        batch_size_per_server = str(max(1, TOTAL_BATCH_SIZE // self.number_of_servers))
        ubatch_size_per_server = str(max(1, TOTAL_UBATCH_SIZE // self.number_of_servers))
        threads_per_server = str(max(1, NUMBER_OF_CORES // self.number_of_servers))
        threads_batch_per_server = str(max(1, TOTAL_THREADS_BATCH // self.number_of_servers))

        command = [
            path,
            "--host", HOST,
            "--port", str(port),
            "--model", GENERAL_MODEL_PATH,
            "--ctx-size", "16000",
            "--repeat-last-n", "0",
            "--gpu-layers", gpu_layers_per_server,
            "--threads", threads_per_server,
            "--threads-batch", threads_batch_per_server,
            "--batch-size", batch_size_per_server,
            "--ubatch-size", ubatch_size_per_server,
            "--dump-kv-cache",
            "--penalize-nl",
            "--seed", "42",
            "--special"
        ]

        try:
            with open(f"llama-server_{port}.log", "w") as log:
                server_process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=log
                )
            await asyncio.sleep(3)  # Simulate time taken to start the server
            print(f"Server started on port {port}")

            # Store the command for future use
            self.server_commands.append(command)

            return server_process
        except Exception as e:
            print(f"Failed to start server on port {port}: {e}")
            return None

    async def spin_up_servers(self):
        """
        Start all configured llama-servers and perform health checks.
        """
        print("Killing any existing processes on configured ports...")
        for port in self.ports:
            kill_process_on_port(port)

        print("Checking if llama-server needs compilation...")
        compile_llama_cpp()

        print("Starting servers...")
        for i in range(self.number_of_servers):
            copy_llama_binary(i)  # Copy llama binary for each server
            server = await self.start_server(f"efs/bin/llama-{i}/llama-server", self.ports[i])
            if server:
                # Health checks
                for attempt in range(5):
                    if is_server_up(self.ports[i]):
                        break
                    print(f"Server {i + 1} not up yet, checking again in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    print(f"Server {i + 1} failed to pass health checks after 5 attempts, exiting...")
                    self.shutdown_servers()
                    raise Exception(f"Server {i + 1} failed to pass health checks.")

                self.servers.append(server)
                self.server_calls.append(0)
            else:
                print(f"Failed to start server {i + 1}, exiting...")
                self.shutdown_servers()
                raise Exception(f"Failed to start server {i + 1}.")

    async def call_llama_server(self, payload):
        """
        Call the active llama-server with the specified payload.
        """
        result = {"error": "No response from server"}
        max_retries = 3  # Set the maximum number of retries
        retry_count = 0

        while retry_count < max_retries:
            with self.lock:
                active_port = self.ports[self.current_server_index]

                if self.switch_in_progress:
                    print("Switch in progress, but continuing with the current active server.")
                else:
                    if self.server_calls[self.current_server_index] >= MAX_CALLS and self.active_requests == 0:
                        print(f"Server {self.current_server_index + 1} reached max calls. Preparing to switch...")
                        self.switch_in_progress = True
                        asyncio.create_task(self.switch_to_standby())

                    self.server_calls[self.current_server_index] += 1
                    self.active_requests += 1

            try:
                async with httpx.AsyncClient(timeout=75) as client:
                    url = f"http://localhost:{active_port}/completion"
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    break

            except httpx.ReadError as e:
                print(f"ReadError occurred: {e}")
                result = {"error": f"ReadError occurred during the request: {e}"}
                await self.wait_for_switch_completion()
                retry_count += 1

            except httpx.HTTPStatusError as e:
                print(f"HTTP error during request: {e}")
                result = {"error": f"Failed to get a valid response from the server: {e}"}
                retry_count += 1

            finally:
                with self.lock:
                    self.active_requests -= 1

                    if self.server_calls[self.current_server_index] >= MAX_CALLS and self.active_requests == 0:
                        if not self.switch_in_progress:
                            print("All requests finished, switching servers.")
                            asyncio.create_task(self.switch_to_standby())

        if retry_count == max_retries:
            print("Max retries reached, returning error result.")
            result = {"error": "Max retries reached, no valid response from server."}

        return result

    async def call_llama_server_stream(self, payload):
        """
        Call the active llama-server with the specified payload.
        Supports streaming responses by setting payload['stream'] to True.
        Yields chunks if streaming, otherwise yields the full result.
        """
        result = {"error": "No response from server"}
        max_retries = 3  # Set the maximum number of retries
        retry_count = 0

        while retry_count < max_retries:
            with self.lock:
                active_port = self.ports[self.current_server_index]

                if self.switch_in_progress:
                    print("Switch in progress, but continuing with the current active server.")
                else:
                    if self.server_calls[self.current_server_index] >= MAX_CALLS and self.active_requests == 0:
                        print(f"Server {self.current_server_index + 1} reached max calls. Preparing to switch...")
                        self.switch_in_progress = True
                        asyncio.create_task(self.switch_to_standby())

                    self.server_calls[self.current_server_index] += 1
                    self.active_requests += 1

            try:
                async with httpx.AsyncClient(timeout=75, follow_redirects=True) as client:
                    url = f"http://localhost:{active_port}/completion"
                    response = await client.post(url, json=payload)
                    async for chunk in response.aiter_text():
                        if chunk:
                            yield chunk
                    break  # Exit the loop after streaming

            except httpx.ReadError as e:
                print(f"ReadError occurred: {e}")
                result = {"error": f"ReadError occurred during the request: {e}"}
                await self.wait_for_switch_completion()
                retry_count += 1

            except httpx.HTTPStatusError as e:
                print(f"HTTP error during request: {e}")
                result = {"error": f"Failed to get a valid response from the server: {e}"}
                retry_count += 1

            finally:
                with self.lock:
                    self.active_requests -= 1

                    if self.server_calls[self.current_server_index] >= MAX_CALLS and self.active_requests == 0:
                        if not self.switch_in_progress:
                            print("All requests finished, switching servers.")
                            asyncio.create_task(self.switch_to_standby())

        if retry_count == max_retries:
            print("Max retries reached, returning error result.")
            yield result

    async def wait_for_switch_completion(self):
        """
        Waits for the standby server to become active during the server switch process.
        """
        while self.switch_in_progress:
            await asyncio.sleep(0.1)
        print("Switch completed, continuing with the new active server.")

    async def switch_to_standby(self):
        """
        Switches the active server to standby and vice versa.
        """
        with self.lock:
            print(
                f"Switching from server {self.current_server_index + 1} to server {self.next_server_index() + 1}...")
            previous_index = self.current_server_index
            self.current_server_index = self.next_server_index()
            self.switch_in_progress = False
            self.server_calls[previous_index] = 0
            print(f"Switched to server {self.current_server_index + 1}.")

    def next_server_index(self):
        """
        Get the next server index in a circular manner.
        """
        return (self.current_server_index + 1) % self.number_of_servers

    def shutdown_servers(self):
        """
        Shut down all llama-servers gracefully.
        """
        print("Shutting down all servers...")
        for server in self.servers:
            if server:
                server.terminate()
        print("All servers have been shut down.")
