ARG BASE_IMAGE=sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base

WORKDIR /src
# clone projects from public/private github repos
RUN git clone https://github.com/sinzlab/neuralpredictors &&\
    git clone https://github.com/sinzlab/nnfabrik

FROM ${BASE_IMAGE}
COPY --from=base /src /src

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip --no-cache-dir install hub

RUN cd /src/neuralpredictors && python3.8 -m pip install --no-use-pep517 -e .
RUN cd /src/nnfabrik && python3.8 -m pip install --no-use-pep517 -e .

ADD . /project
RUN python3.8 -m pip install -e /project
WORKDIR /notebooks

