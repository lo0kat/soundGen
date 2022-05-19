FROM python:3.8

# 
WORKDIR /deployment

RUN apt-get update && apt-get install -y libsndfile1 

# 
COPY ./requirements.txt /deployment/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /deployment/requirements.txt

# 
COPY ./app /deployment/app

# 
 CMD ["python3","app/my_deployment.py"]