FROM condaforge/miniforge3:latest
ENV DEBIAN_FRONTEND=noninteractive PATH=/opt/conda/bin:$PATH PYTHONPATH=/home:$PYTHONPATH

RUN printf '%s\n' \
    '#!/bin/bash' \
    'conda run --no-capture-output -n aimnet2 jupyter "$@"' \
    > /usr/bin/jupyter && \
    chmod +x /usr/bin/jupyter

WORKDIR /home

COPY environment.yml environment-mace.yml environment-uma.yml ./
RUN apt-get update && apt-get -y install --no-install-recommends git gcc g++ && \
    mamba env create -f environment.yml && \
    mamba env create -f environment-mace.yml && \
    mamba env create -f environment-uma.yml && \
    mamba clean --all -afy && \
    find /opt/conda -follow -type f -name '*.a' -delete && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    find /opt/conda -follow -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Cache the default AIMNet2 weights outside user home directories.
ENV AIMNET_CACHE_DIR=/opt/aimnet-models
RUN /opt/conda/envs/aimnet2/bin/python -c \
    "from aimnet.calculators.model_registry import get_model_path; print(get_model_path('aimnet2'))" && \
    chmod -R a+rX "$AIMNET_CACHE_DIR"

ARG CACHEBUST
COPY cli.py ./
COPY kernels /usr/local/share/jupyter/kernels/
COPY demo_tools /home/demo_tools
COPY demo /home/demo

ENTRYPOINT ["python", "/home/cli.py"]
