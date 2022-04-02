FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
RUN apt-get update -y

# Install Python3.8 (for Ubuntu20.04)
RUN apt-get install -y python3 python3-pip python3-venv

# Set up PyEnv
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install Python Dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
