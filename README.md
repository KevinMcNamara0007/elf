##### REPO_LINK AND BRANCH -> https://github.com/KevinMcNamara0007/elf/tree/chroma_addition

## To Install Locally from source
#### Linux
###### Requires Python 3.10
1. sudo apt-get install cmake git libopenblas-dev pkg-config
2. sudo python3 -m pip install -r requirements.txt
   1. #### To Enable GPU support
      1. sudo python3 -m pip install tensorflow==2.15.0 keras==2.15.0
3. Create a directory called "efs" in the project root folder
4. Download gguf files of models you want to use and place them in efs/models under root folder
   1. Ideally use https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF -> Q4_K_M is preferred
5. Include your model path under the designated config/.env-[environment] variables


## To Start Locally from source (Linux)
sudo uvicorn src.asgi:elf --reload --host=127.0.0.1 --port==8080 --env-file confg/.env-dev

## Running from Image (Locally or Cloud)
### Options
1. docker pull darkiroha/containerized-gas-cuda
2. docker pull darkiroha/containerized-gas-cpu
##### The files inside the directory "efs/classifier" should be in your efs storage under the path "/app/efs/classifier"
##### failure to do so will cause in errors at startup.
##### If using this image locally then you can mount the path of those files onto your docker container using the "/app/efs" path as the 