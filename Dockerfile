FROM python:3.10.1-buster

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER nasr@l3s.de

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

RUN pip install  pandas==1.3.4
RUN pip install  tensorflow==2.8.0
RUN pip install  librosa==0.9.1
RUN pip install  Numpy==1.21.4
RUN pip install  scipy==1.7.3
RUN pip install  scikit-learn==1.0.2
RUN pip install  joblib==1.1.0
RUN pip install SoundFile

## Include the following line if you have a requirements.txt file.
#RUN pip install -r requirements.txt
