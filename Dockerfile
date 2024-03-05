FROM python:3.8.8-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
       git \
       libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir jupyterlab

RUN rm -rf requirements.txt

EXPOSE 8888