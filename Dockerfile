# syntax=docker/dockerfile:1

FROM python:3.7.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "main.py"]