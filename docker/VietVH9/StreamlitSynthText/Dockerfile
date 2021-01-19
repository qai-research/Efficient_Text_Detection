FROM ubuntu:18.04

 

###################################################

 

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get update

 

###################################################

 

WORKDIR /work 
COPY ./ocr-components/akaocr/requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install streamlit

 

################################################### 

COPY ./data ./data
COPY ./ocr-components/akaocr ./code/akaocr
COPY ./ocr-components/streamlitapp/ ./code/streamlitapp 

RUN cd ./code/streamlitapp && streamlit run SynthText_app.py