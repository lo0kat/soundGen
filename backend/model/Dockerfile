FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt update ; apt install -y ffmpeg 

WORKDIR /model

# 
COPY ./requirements.txt /model/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /model/requirements.txt

WORKDIR /model/train_model