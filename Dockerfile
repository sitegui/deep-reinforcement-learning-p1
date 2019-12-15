FROM continuumio/miniconda3:4.7.10

WORKDIR /app

# Install Unity environment
RUN apt-get update -y && \
    apt install -y unzip && \
    wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip && \
    unzip Tennis_Linux_NoVis.zip

# Main conda dependencies
COPY environment.yml .
RUN conda env create
RUN echo "conda activate deep-reinforcement-learning" >> ~/.bashrc
SHELL [ "/bin/bash", "-l", "-c" ]
ENV PATH=/opt/conda/envs/deep-reinforcement-learning/bin:$PATH

# PyTorch
RUN conda install pytorch torchvision cpuonly -c pytorch

# Ml-agents
RUN git clone https://github.com/udacity/deep-reinforcement-learning.git && \
    cd deep-reinforcement-learning/python && \
    pip install .

# Main app
COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT [ "/opt/conda/envs/deep-reinforcement-learning/bin/python",  "train.py" ]

