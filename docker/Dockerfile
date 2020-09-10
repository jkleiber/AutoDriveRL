FROM ubuntu:bionic

# Update linux repositories.
RUN apt-get update

# Install dependencies
RUN apt-get install -y python3-pip wget
RUN apt-get install -y git

# Install conda dependencies for simulator.
WORKDIR /setup
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
ENV PATH=/miniconda/bin:${PATH}

# Change to the project workspace's third party software directory.
WORKDIR /workspace/third_party

# Clone the donkey repository.
RUN git clone https://github.com/autorope/donkeycar
WORKDIR /workspace/third_party/donkeycar
RUN git checkout master

# Create a conda environment. Activate conda by remapping the bash commands to the conda subprocess.
RUN conda env create -f install/envs/ubuntu.yml
SHELL ["conda", "run", "-n", "donkey", "/bin/bash", "-c"]
RUN pip install -e .[pc]

# Create the donkey car gym.
WORKDIR /workspace/third_party/
RUN git clone https://github.com/tawnkramer/gym-donkeycar
WORKDIR /workspace/third_party/gym-donkeycar
RUN pip install -e .[gym-donkeycar]

# Change back to project root.
WORKDIR /workspace

# Expose simulator ports
EXPOSE 9091
EXPOSE 8887

CMD tail -f /dev/null