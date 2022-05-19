FROM python:3.8.10-slim

# 
WORKDIR /frontend

RUN apt-get update && apt-get install -y ffmpeg 

# 
COPY ./requirements.txt /frontend/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /frontend/requirements.txt

# 
COPY ./app /frontend/app

# 
 CMD ["python3","app/app.py"]