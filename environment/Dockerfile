# hash:sha256:396cc81e4044e6881d210773083a9db2380e5b56c7ca51cc9463c74d6a3ab95c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    aind-log-utils==0.2.3 \
    dartsort==0.5.26 \
    scipy==1.17.1 \
    spikeinterface[full]==0.104.7
