FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip --no-cache-dir install hub git+https://github.com/sinzlab/neuralpredictors.git@main

COPY ./cascade /project/cascade
COPY ./setup.py /project
COPY ./eval.py /project
COPY ./test_eval.py /project

WORKDIR /project

RUN python3.8 -m pip install -e .
