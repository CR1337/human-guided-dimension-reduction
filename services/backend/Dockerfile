FROM python:3.10

WORKDIR /server

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /server/

RUN chmod +x /server/compile-neighbors2d
RUN ./compile-neighbors2d

CMD ["python3", "server.py"]
