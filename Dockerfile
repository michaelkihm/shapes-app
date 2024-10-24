FROM python:3.12-slim-bullseye

#RUN apt update && apt install -y libc-dev libffi-dev libjpeg-dev ffmpeg gcc git zlib1g g++ curl libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev

WORKDIR /shapes

COPY requirements/base.txt .
RUN pip install -U pip
RUN pip install --no-cache-dir -r base.txt