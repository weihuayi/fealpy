
FROM python:3.8-slim-buster

WORKDIR /data/fealpy
ADD requirements.txt ./
RUN pip install -r requirements.txt
COPY ./ ./
RUN pip install .
