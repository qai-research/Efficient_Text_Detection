FROM python:3.7

COPY akaocr/requirements.txt .
RUN pwd
RUN pip install -r requirements.txt && rm -rf requirements.txt
RUN pip install dvc
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get update && \
      apt-get upgrade -y && \
      apt-get install -y git
RUN mkdir /app
WORKDIR /app
RUN python --version
CMD /bin/bash
