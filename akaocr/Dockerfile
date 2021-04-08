FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN pwd
RUN ls
COPY /requirements.txt .
RUN ls
RUN pip install -r requirements.txt && rm -rf requirements.txt
RUN pip install dvc

RUN mkdir /app
WORKDIR /app

CMD /bin/bash
