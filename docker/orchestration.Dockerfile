FROM python:3.11-slim

ARG SOURCE_REVISION
WORKDIR /opt/solarcast
ENV CI=true \
    DVC_NO_ANALYTICS=true \
    DVC_SITE_CACHE_DIR=/tmp/dvc-site \
    AWS_EC2_METADATA_DISABLED=true \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN test "${#SOURCE_REVISION}" -eq 40 \
    && git init \
    && git fetch --depth=1 https://github.com/AwanTech095/SolarCast.git "$SOURCE_REVISION" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$SOURCE_REVISION"
RUN python -m pip install --no-cache-dir -r requirements-ci.txt \
    && python -m venv /opt/dvc \
    && /opt/dvc/bin/python -m pip install --no-cache-dir -r requirements-dvc.txt
