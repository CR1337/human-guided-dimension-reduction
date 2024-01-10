FROM python:3.10

WORKDIR /server

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /server/

CMD ["python3", "server.py"]
