FROM python:3.7
MAINTAINER huukim98@gmail.com(kimnh3)

COPY akaocr/requirements.txt .
RUN pwd
RUN pip install -r requirements.txt && rm -rf requirements.txt
RUN pip install dvc dvc[ssh]
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir /app
CMD /bin/bash
