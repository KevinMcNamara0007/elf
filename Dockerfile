# Use Python base image
FROM python:3.10-slim-bookworm AS env-build

# Install build tools and dependencies, including Tcl
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    cmake \
    wget \
    python3-dev \
    unzip \
    tcl

# Download official SQLite amalgamation files (version 3460100)
WORKDIR /opt/sqlite
RUN wget https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip && \
    unzip sqlite-amalgamation-3460100.zip

# Clone the pysqlite3 repository
RUN git clone https://github.com/coleifer/pysqlite3.git /opt/cx_intelligence/aiaas/pysqlite3

# Copy SQLite amalgamation files into pysqlite3
RUN cp /opt/sqlite/sqlite-amalgamation-3460100/sqlite3.[ch] /opt/cx_intelligence/aiaas/pysqlite3/

# Build pysqlite3 statically linked with SQLite
WORKDIR /opt/cx_intelligence/aiaas/pysqlite3
RUN python setup.py build_static build

RUN pip install keras==2.15.0
RUN pip install tensorflow==2.15.0

# Switch to the application build stage
FROM env-build AS app

# Set the working directory to /app
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Expose necessary ports
EXPOSE 8000-8010

# Set the entry point for FastAPI app
CMD ["uvicorn", "src.asgi:elf", "--host=0.0.0.0", "--port=8000"]
