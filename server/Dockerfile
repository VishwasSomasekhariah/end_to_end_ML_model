FROM python:latest

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ..

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD python server.py