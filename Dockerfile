FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7
WORKDIR /src
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip --no-cache-dir install hub
ADD . /project
RUN python3.8 -m pip install -e /project
WORKDIR /notebooks

