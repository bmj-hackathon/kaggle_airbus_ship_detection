FROM python:3.8.5

LABEL maintainer="batman@batman.com"

COPY ./requirements.txt .
COPY ./dash_app ./dash_app
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-dev

CMD python ./dash_app/app.py --data_volume "/data"
