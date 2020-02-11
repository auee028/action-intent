FROM nvidia/cuda:8.0-cudnn6-devel
MAINTAINER abr <youhs4554@gmail.com>

#RUN apt-get update
RUN apt-get update && apt-get install -y build-essential \
    checkinstall \
	gcc \
	git \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
	software-properties-common \
	tk-dev \
    tar \
    wget \
    zip \
	python2.7 python2.7-dev python-pip python-setuptools python-wheel
	
WORKDIR /tmp

# install fundamental packages including python
#RUN add-apt-repository -y ppa:fkrull/deadsnakes
#RUN apt-get update
#RUN apt-get install -y --no-install-recommends python2.7 python2.7-dev python-pip python-setuptools python-wheel gcc

# pip 업그레이드
RUN python -m pip install pip --upgrade

WORKDIR /
RUN rm -rf /tmp

RUN mkdir /src
COPY vintent.tar.gz /src

WORKDIR /src

RUN tar -xvzf vintent.tar.gz --strip-components=1

#RUN pip install -r requirements.txt -i http://ftp.daumkakao.com/pypi/simple --trusted-host ftp.daumkakao.com
RUN pip install -r requirements.txt

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

#RUN python server.py
CMD ["python", "server.py"]

EXPOSE 50052

