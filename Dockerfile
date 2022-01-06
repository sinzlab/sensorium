ARG BASE_IMAGE=sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src

# Use git credential-store to specify username and pass to use for pulling repo
RUN git config --global credential.helper store &&\
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials

# clone projects from public/private github repos
RUN git clone --depth 1 --branch challenge https://github.com/${DEV_SOURCE}/data_port.git

FROM ${BASE_IMAGE}
COPY --from=base /src /src

RUN cd /src/data_port && python setup.py develop

COPY ./neuralpredictors /src/neuralpredictors
RUN cd /src/neuralpredictors && python setup.py develop

WORKDIR /project
RUN mkdir /project/cascade
COPY ./neural-prediction-challenge/cascade /project/cascade
COPY ./neural-prediction-challenge/setup.py /project
RUN python setup.py develop
