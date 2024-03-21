[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/CR1337/human-guided-dimension-reduction/blob/main/LICENSE)

# human-guided-dimension-reduction

A web application for experimenting with dimensionality reduction with human guidance.

## Setup

### Linux

1. Install [Docker](https://docs.docker.com/desktop/install/linux-install/)
2. Clone this Repository using `git clone https://github.com/CR1337/human-guided-dimension-reduction.git`.
3. cd into this repository with `cd human-guided-dimension-reduction`.
4. Build the docker services by running `bin/build`.
5. Create a virtual environment by executing `python3 -m venv .venv`.
6. Activate the virtual environment with `source .venv/bin/activate`.
7. Install dependencies using `pip3 install -r requirements.txt`.
8. Change directory with `cd services/backend`.
9. Run `./compile-neighbors`.
10. Run `chmod +x neighbors/neighbors`.
11. Change back to the main directory with `cd ../..`.
12. Run `python3 util/init_neighbors.py`.

### Windows
_(Not tested, you are on your own.)_

1. Install [Docker](https://docs.docker.com/desktop/install/windows-install/)
2. Clone this Repository using `git clone https://github.com/CR1337/human-guided-dimension-reduction.git`.
3. cd into this repository with `cd human-guided-dimension-reduction`.
4. Build the docker services by running `bin\build.bat`.
5. Create a virtual environment by executing `python -m venv .venv`.
6. Activate the virtual environment with `.venv\bin\activate`.
7. Install dependencies using `pip install -r requirements.txt`.
8. Make sure you have g++ installed (at least version 11.4).
9. Change directory with `cd services\backend`.
10. Run `./compile-neighbors.bat`.
11. Change back to the main directory with `cd ..\..`.
12. Run `python3 util/init_neighbors.py`.

## Running the App

### Linux

1. Execute `bin/run`
2. Open your Browser and navigate to [http://localhost:8080](http://localhost:8080)

### Windows

1. Execute `bin\run.bat`
2. Open your Browser and navigate to [http://localhost:8080](http://localhost:8080)
