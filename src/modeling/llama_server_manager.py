import os
import subprocess
import threading
import asyncio
import httpx
import psutil
from fastapi import HTTPException

LLAMA_PORT = int(os.getenv("LLAMA_PORT", 8001))
GENERAL_MODEL_PATH = os.getenv("general", "efs/models/Llama-3.1.gguf")
LLAMA_CPP_HOME = os.getenv("LLAMA_CPP_HOME", "/opt/cx_intelligence/aiaas/compiled_llama_cpp")
LLAMA_CPP_PATH = os.path.join(LLAMA_CPP_HOME, "bin/llama-server")
HOST = os.getenv("HOST", "0.0.0.0")
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "99"))
NUMBER_OF_CORES = os.cpu_count()
TOTAL_BATCH_SIZE = int(os.getenv("TOTAL_BATCH_SIZE", "8196"))
TOTAL_UBATCH_SIZE = int(os.getenv("TOTAL_UBATCH_SIZE", "2048"))
TOTAL_THREADS_BATCH = int(os.getenv("TOTAL_THREADS_BATCH", "4"))
MAX_CALLS = int(os.getenv("MAX_CALLS", "35"))


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
        self.current_server_index = 0
        self.active_requests = 0
        self.lock = threading.Lock()
        self.switch_in_progress = False
        self.is_request_in_progress = [False] * self.number_of_servers  # Track if a request is being processed

        # Initialize ports and active server indices
        self.ports = [LLAMA_PORT + i for i in range(self.number_of_servers)]
        self.active_server_indices = list(range(self.number_of_servers - 1))  # All but the last one are active
        self.standby_server_index = self.number_of_servers - 1  # Last server is standby

    async def start_server(self, path, port, index):
        """
        Start a llama-server instance with the specified resource configuration.
        """
        self.kill_process_on_port(port)
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
            print(f"Server {index + 1} started on port {port}")
            return server_process
        except Exception as e:
            print(f"Failed to start server on port {port}: {e}")
            return None

    def kill_process_on_port(self, port):
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

    async def spin_up_servers(self):
        """
        Start all configured llama-servers and perform health checks.
        """
        print("Starting servers...")
        for i in range(self.number_of_servers):
            server = await self.start_server(LLAMA_CPP_PATH, self.ports[i], i)
            if server:
                # Health checks
                for attempt in range(5):
                    if self.is_server_up(self.ports[i]):
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

    def is_server_up(self, port):
        """
        Check if the server is up by sending a simple request.
        """
        url = f"http://localhost:{port}/health"
        try:
            response = httpx.get(url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    async def call_llama_server(self, payload):
        """
        Calls the active llama-server and ensures that no requests are skipped during server transitions.
        """
        result = {"error": "No response from server"}  # Initialize result with a default value

        while True:  # Keep trying until we get a valid active server
            with self.lock:
                active_port = self.ports[self.current_server_index]

                if self.server_calls[self.current_server_index] >= MAX_CALLS:
                    # Prevent new requests during switch
                    if self.switch_in_progress:
                        print("Switch in progress, waiting for standby to become active...")
                        await asyncio.sleep(1)
                        continue  # Wait for switch to complete, then retry

                    # Redirect to standby server and start switch
                    self.current_server_index = self.standby_server_index
                    active_port = self.ports[self.current_server_index]

                self.server_calls[self.current_server_index] += 1
                call_count = self.server_calls[self.current_server_index]
                self.active_requests += 1
                self.is_request_in_progress[self.current_server_index] = True  # Mark request in progress

            url = f"http://localhost:{active_port}/completion"

            try:
                async with httpx.AsyncClient(timeout=75) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()  # Raise an error for bad responses
                    result = response.json()  # Set result here if the request is successful
                    break  # Exit the loop when the request succeeds

            except httpx.HTTPStatusError as e:
                print(f"Error in llama-server request: {e}")
                result = {"error": f"Failed to get a valid response from the server: {e}"}  # Handle error

            finally:
                with self.lock:
                    self.active_requests -= 1
                    self.is_request_in_progress[self.current_server_index] = False  # Mark request as complete

                    # Check if the server should switch after this request
                    if call_count >= MAX_CALLS and self.active_requests == 0 and not self.is_request_in_progress[
                        self.current_server_index]:
                        print(
                            f"Server {self.current_server_index + 1} reached max calls ({call_count}). Switching servers...")
                        asyncio.create_task(self.switch_to_standby())  # Trigger server switch

        return result

    async def call_llama_server_stream(self, payload):
        """
        Calls the active llama-server at the /completion endpoint and increments the call count.
        If the server exceeds the max call count, it triggers a server switch asynchronously after the current request completes.
        This streams out the call chunks.
        """
        with self.lock:
            active_port = self.ports[self.current_server_index]
            self.server_calls[self.current_server_index] += 1
            call_count = self.server_calls[self.current_server_index]
            self.active_requests += 1
            self.is_request_in_progress[self.current_server_index] = True  # Mark request in progress

        url = f"http://localhost:{active_port}/completion"

        try:
            async with httpx.AsyncClient(timeout=75) as client:
                async with client.stream("POST", url, json=payload) as response:
                    async for chunk in response.aiter_text():
                        yield chunk

        except httpx.RequestError as e:
            print("Network Error", str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            print(f"Response Error", str(e))
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            with self.lock:
                self.active_requests -= 1
                self.is_request_in_progress[self.current_server_index] = False  # Mark request as complete

                # Check if the server should switch after this request
                if call_count >= MAX_CALLS and self.active_requests == 0 and not self.is_request_in_progress[
                    self.current_server_index]:
                    print(
                        f"Server {self.current_server_index + 1} reached max calls ({call_count}). Switching servers...")
                    asyncio.create_task(self.switch_to_standby())  # Trigger server switch after the request is done

    async def switch_to_standby(self):
        """
        Switches the active server to standby and restarts it.
        The previous standby server becomes the new active server.
        """
        with self.lock:
            if self.switch_in_progress:
                print("Switch in progress, skipping...")
                return
            self.switch_in_progress = True

        try:
            # Shut down the current server
            server_index = self.current_server_index
            server = self.servers[server_index]
            print(f"Restarting server {server_index + 1} on port {self.ports[server_index]}...")
            server.terminate()
            await asyncio.sleep(1)  # Simulate some time to shut down the server
            server.wait()

            # Start the server again as a new standby
            new_server_process = await self.start_server(LLAMA_CPP_PATH, self.ports[server_index], server_index)
            if new_server_process:
                self.servers[server_index] = new_server_process
                print(f"Server {server_index + 1} restarted and ready as standby.")

                # Swap active and standby servers
                self.current_server_index = (self.standby_server_index)
                self.standby_server_index = server_index  # Update standby index

                print(f"Active server switched to server {self.current_server_index + 1}.")

            else:
                print(f"Failed to restart server {server_index + 1}.")

        finally:
            with self.lock:
                self.switch_in_progress = False

    def shutdown_servers(self):
        """
        Shut down all running llama-servers.
        """
        for i, server in enumerate(self.servers):
            if server:
                print(f"Shutting down server {i + 1} on port {self.ports[i]}...")
                server.terminate()
                server.wait()
                print(f"Server {i + 1} shut down successfully.")
        self.servers.clear()
        self.server_calls.clear()
