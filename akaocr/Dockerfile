FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

COPY akaocr/requirements.txt .
RUN pwd
RUN pip install -r requirements.txt && rm -rf requirements.txt
RUN pip install dvc

RUN mkdir /app
WORKDIR /app
RUN python --version
CMD /bin/bash
