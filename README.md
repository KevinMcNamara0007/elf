## To Install
#### Linux
Using conda:
1. conda create -n <virtual env name> => This is optional
2. conda activate <virtual env name> => This is optional
3. sudo apt install pytesseract-ocr
   1. which tesseract
   2. copy the path this prints
   3. export PATH="<copied path>:$PATH"
4. pip install -r requirements.txt
   1. #### To Enable GPU support
      1. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 => Run inside of venv or on container
5. Download gguf files of models you want to use and place them in efs/models under root folder

## To Start
uvicorn src.asgi:elf --reload --host=127.0.0.1 --port==8080