FROM python:3.8.5

LABEL maintainer="batman@batman.com"

COPY ./requirements.txt .

RUN pip install -r requirements.txt


