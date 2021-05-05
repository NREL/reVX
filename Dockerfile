# syntax=docker/dockerfile:1
FROM continuumio/miniconda3

WORKDIR /reVX
RUN mkdir -p /reVX
RUN conda update conda -y && \
    conda create --name revx --yes python=3.8
# Copy package
COPY . /reVX

# Install dependencies
RUN conda install --name revx --yes \
    pip \
    git \
    rtree \
    geopandas \
    && conda run -n revx --no-capture-output pip install --no-cache-dir .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "revx", "reVX"]
