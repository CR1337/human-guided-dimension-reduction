FROM python:3.10

WORKDIR /server

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /server/

RUN chmod +x /server/compile-neighbors
RUN ./compile-neighbors
RUN chmod +x /server/neighbors/neighbors

CMD ["python3", "server.py"]
