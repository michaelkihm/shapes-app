FROM python:3.12-slim-bullseye

RUN apt-get update && apt-get install -y libc-dev libffi-dev libjpeg-dev ffmpeg gcc

WORKDIR /shapes

COPY requirements/base.txt .
RUN pip install -U pip
RUN pip install --no-cache-dir -r base.txt

COPY shapes /shapes