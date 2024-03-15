## To Install
#### Linux
Using conda:
1. conda create -n <virtual env name>
2. conda activate <virtual env name>
3. pip install -r requirements.txt
4. download gguf files of models you want to use and place them in efs/models under root folder

## To Start
uvicorn src.asgi:elf --reload --host=127.0.0.1 --port==8080