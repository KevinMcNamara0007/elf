## To Install
#### MacOS
###### Requires Python 3.9
1. brew install git-lfs
2. git lfs install
3. pip install -r requirements.txt
   1. #### To Enable GPU support
      1. python3 install tensorflow==2.15.0 keras==2.15.0
4. Download gguf files of models you want to use and place them in efs/models under root folder
5. Include your model path under the designated config/.env-[environment] variables

## To Start
sudo uvicorn src.asgi:elf --reload --host=127.0.0.1 --port==8080