FROM continuumio/miniconda3
ADD dlenv.yml dlenv.yml
RUN conda env create -f dlenv.yml

# Pull the environment name out of the dlenv.yml
RUN echo "source activate $(head -1 dlenv.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 dlenv.yml | cut -d' ' -f2)/bin:$PATH
