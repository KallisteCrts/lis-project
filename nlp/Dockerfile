FROM ubuntu:18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8 

RUN apt-get update
RUN apt install -y build-essential 
RUN apt install -y bash 
RUN apt install -y git 
RUN apt install -y curl 
RUN apt install -y python3 
RUN apt install -y python3-pip

RUN mkdir /app
WORKDIR /app

ADD requirements.txt /app
ADD /src/main.py /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip install torch torchvision --no-cache-dir
RUN pip install sentence-transformers --no-cache-dir
RUN pip install nltk --no-cache-dir

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]