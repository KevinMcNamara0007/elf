##### REPO_LINK AND BRANCH -> https://github.com/KevinMcNamara0007/elf/tree/chroma_addition

## To Install Locally from source
#### MacOS
###### Requires Python 3.9
1. brew install cmake
   1. Add cmake to path
      1. brew --prefix cmake
      2. nano ~/.bash_profile or nano ~/.zshrc
      3. Add the following line to the file
         1. Intel Based Mac
            1. export PATH="/usr/local/bin:$PATH"
         2. Silicon Based Mac
            1. export PATH="/opt/hombrew/bin:$PATH"
      4. source ~/.bash_profile or source ~/.zshrc
2. pip install -r requirements-mac.txt
3. pip install -r requirements-app.txt
   1. Add chromadb to path
      1. python3 -m site --user-base
      2. nano ~/.bash_profile or nano ~/.zshrc
      3. export PATH="$HOME/.local/bin:$PATH"
      4. source ~/.bash_profile or source ~/.zshrc
4. Verify both installations:
   1. cmake --version
   2. chromadb --help
5. Copy and paste the directory called "efs" from the zip file into the project root folder
6. Download gguf files of models you want to use and place them in efs/models under root folder
   1. Ideally use https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF -> Q4_K_M is preferred
7. Include your model path as the general env variable in the config/.env-[environment]

#### Linux
###### Requires Python 3.9
1. apt-get install cmake
2. python3 -m pip install -r requirements-app.txt
   1. #### To Enable GPU support
      1. python3 -m pip install tensorflow==2.15.0 keras==2.15.0
3. Create a directory called "efs" in the project root folder
4. Download gguf files of models you want to use and place them in efs/models under root folder
   1. Ideally use https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF -> Q4_K_M is preferred
5. Include your model path under the designated config/.env-[environment] variables


## To Start Locally from source (MAC and Linux)
sudo uvicorn src.asgi:elf --reload --host=127.0.0.1 --port==8080 --env-file confg/.env-dev

## Running from Image (Locally or Cloud)
### Options
1. docker pull darkiroha/containerized-gas-cuda
2. docker pull darkiroha/containerized-gas-cpu
##### The files inside the directory "efs/classifier" should be in your efs storage under the path "/app/efs/classifier"
##### failure to do so will cause in errors at startup.
##### If using this image locally then you can mount the path of those files onto your docker container using the "/app/efs" path as the 