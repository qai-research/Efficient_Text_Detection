FROM pytorch/pytorch:latest

COPY akaocr/requirements.txt .
RUN pwd
RUN pip install -r requirements.txt && rm -rf requirements.txt
RUN pip install dvc

RUN mkdir /app
WORKDIR /app
RUN python --version
CMD /bin/bash
