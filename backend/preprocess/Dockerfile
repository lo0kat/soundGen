FROM python:3.8.10-slim

# 
WORKDIR /preprocess


RUN apt-get update && apt-get install -y libsndfile1 ffmpeg 

# 
COPY ./requirements.txt /preprocess/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /preprocess/requirements.txt

# 
COPY ./app /preprocess/app

# 
 CMD ["python3","app/main.py"]