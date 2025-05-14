# hash:sha256:968bd1ef3dcdd810eeea6314ae26427c267777c61e69b5f3c700b1cf903abf25
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.18_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.102.3
