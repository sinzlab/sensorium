ARG BASE_IMAGE=sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE=sinzlab
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src
# Use git credential-store to specify username and pass to use for pulling repo

RUN git clone https://github.com/sinzlab/neuralpredictors &&\
    git clone https://github.com/sinzlab/nnfabrik &&\

FROM ${BASE_IMAGE}
COPY --from=base /src /src
ADD . /src/cascade

RUN pip install -e /src/neuralpredictors &&\
    pip install -e /src/nnfabrik &&\
    pip install -e /src/nndichromacy &&\
    pip install -e /src/mei &&\
    pip install -e /src/data_port &&\
    pip install -e /src/nexport

