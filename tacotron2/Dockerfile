FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update -y

RUN pip install protobuf==3.19.6
RUN pip install \
    matplotlib==2.1.0 \
    tensorflow==1.15.2 \
    numpy==1.17.0 \
    inflect==0.2.5 \
    librosa==0.6.0 \
    scipy==1.0.0 \
    Unidecode==1.0.22 \
    pillow

RUN pip install tensorboardX  jupyter

ADD apex /apex/
WORKDIR /apex/
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
