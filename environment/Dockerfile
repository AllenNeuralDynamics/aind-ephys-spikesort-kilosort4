# hash:sha256:963c720bf29502d41d6550d0cf780a59f64a06542580c6283d708e3777fe52d4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.12_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1 \
    spikeinterface[full]==0.100.8
