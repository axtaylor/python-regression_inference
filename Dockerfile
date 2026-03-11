FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y    \
    python3                                 \
    python3-pip                             \
    python3.10-venv                         

COPY . .

RUN pip3 install --no-cache-dir  \
    -r requirements.txt          

RUN python3 -m build && \
    pip3 install --no-cache-dir ./dist/*.whl