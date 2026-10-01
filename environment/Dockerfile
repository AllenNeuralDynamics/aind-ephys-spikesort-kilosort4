# hash:sha256:3535001eb2fa6896c9d86b75832ea85e84b36918ae384536fce6e36a80b91ef1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-log-utils==0.2.3 \
    log-schema==0.2.10.dev1 \
    scipy==1.17.1 \
    spikeinterface[full]==0.105.0
