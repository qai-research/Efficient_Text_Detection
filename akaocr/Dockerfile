FROM pytorch/pytorch:latest

RUN pwd
RUN ls
COPY akaocr/requirements.txt .
RUN ls
RUN pip install -r requirements.txt && rm -rf requirements.txt
RUN pip install dvc

RUN mkdir /app
WORKDIR /app
RUN python --version
CMD /bin/bash
