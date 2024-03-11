FROM python:3.10

WORKDIR /server

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./compile-neighbors /server/
COPY ./neighbors /server/neighbors

RUN chmod +x /server/compile-neighbors
RUN ./compile-neighbors
RUN chmod +x /server/neighbors/neighbors

COPY . /server/.

ENV INSIDE_DOCKER=1

CMD ["python3", "server.py"]
