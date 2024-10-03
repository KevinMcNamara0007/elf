import os
import subprocess
import chromadb
import psutil
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import time


class ChromaServerManager:
    def __init__(
            self,
            chroma_db_path=os.getenv("CHROMA_DATA_PATH"),
            chroma_port=os.getenv("CHROMA_PORT", "8000")  # Updated port to match your config
    ):
        """
        Initializes the ChromaServerManager with the ChromaDB path, port, and HTTP client.
        """
        self.chroma_db_path = chroma_db_path
        self.chroma_port = chroma_port
        self.chroma_server_pid = None
        self.client = None
        self.host = os.getenv("HOST", "localhost")  # Updated to match your config
        self.cwd = os.getcwd()

    def start_chroma_db(self):
        """
        Starts the ChromaDB server and verifies its startup using the heartbeat function.
        """
        command = [
            "chroma",
            "run",
            "--host", self.host,
            "--port", str(self.chroma_port),
        ]
        try:
            os.makedirs(self.chroma_db_path, exist_ok=True)
            os.chdir(self.chroma_db_path)

            # Start the ChromaDB server
            self.chroma_server_pid = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait for the server to start
            time.sleep(2)  # Give the server some time to start

            # Initialize the ChromaDB HTTP client
            self.client = chromadb.HttpClient(
                host='localhost',
                port=int(self.chroma_port),
                ssl=False,
                headers=None,
                settings=Settings(),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )

            # Check the heartbeat of the ChromaDB server
            if not self.check_heartbeat():
                error_msg = self.chroma_server_pid.communicate()[1].decode('utf-8')
                print(f"Failed to start ChromaDB server.\n\nERROR:\n\n{error_msg}")
                self.shutdown_chroma_db()
                exit(1)

            print(f"ChromaDB server started successfully on port {self.chroma_port}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start ChromaDB server: {e}")
            exit(1)
        finally:
            os.chdir(self.cwd)

    def check_heartbeat(self):
        """
        Checks if the ChromaDB server is alive using the heartbeat function.
        Attempts the check up to 5 times with a 4-second wait between each attempt.
        """
        for attempt in range(5):
            try:
                response = self.client.heartbeat()
                # Check if response is an integer indicating server health
                if isinstance(response, int) and response >= 0:
                    return True  # Server is alive
                else:
                    print(f"Heartbeat check attempt {attempt + 1} returned an invalid response: {response}")
            except Exception as e:
                print(f"Heartbeat check attempt {attempt + 1} failed: {e}")
            time.sleep(4)  # Wait 4 seconds before the next attempt
        return False  # Server did not respond after 5 attempts

    def shutdown_chroma_db(self):
        """
        Shuts down the ChromaDB server.
        """
        for conn in psutil.net_connections():
            if conn.laddr.port == self.chroma_port:
                pid = conn.pid
                if pid:
                    try:
                        os.kill(pid, 9)
                        print(f"Process {pid} on port {self.chroma_port} has been killed.")
                    except Exception as e:
                        print(f"Failed to kill process {pid} on port {self.chroma_port}: {e}")
                else:
                    print(f"No process found running on port {self.chroma_port}.")
                return
        print(f"No process found running on port {self.chroma_port}.")
