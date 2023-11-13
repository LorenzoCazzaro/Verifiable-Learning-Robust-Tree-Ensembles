FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt-get install software-properties-common -y
RUN apt update -y && apt-get upgrade -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update -y
RUN apt install wget
RUN apt install xz-utils
RUN apt-get install cmake -y
RUN apt-get install python3.8 -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
RUN update-alternatives --set python3 /usr/bin/python3.8
RUN apt install python3.8-distutils -y
RUN apt-get install python3-pip -y

WORKDIR /home

RUN mkdir /home/src
RUN mkdir /home/datasets

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

COPY src/ /home/src

COPY datasets/ /home/datasets

COPY setup.sh /home

RUN apt install -y nano

CMD ["bash"]
