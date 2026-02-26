FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

# User Arguments
ARG USERNAME=plb41586
ARG USER_UID=1996
ARG USER_GID=1996

# Install Sudo
RUN apt-get update
RUN apt-get install -y sudo

# Create the user and group
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set permissions on home directory
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

RUN apt-get install -y build-essential \
    curl \
    git \
    llvm \
    clang \
    libclang-dev \
    libssl-dev \
    neovim \
    libpcap-dev

# Switch to non-root by default if desired
USER $USERNAME

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

USER root

RUN apt-get -y install python3.12-venv
RUN python -m venv /home/$USERNAME/app/venv
ENV VIRTUAL_ENV=/home/$USERNAME/app/venv/
ENV PATH=/home/$USERNAME/app/venv/bin:$PATH
RUN /home/$USERNAME/app/venv/bin/pip install "redis[hiredis]"
RUN /home/$USERNAME/app/venv/bin/pip install msgpack
RUN /home/$USERNAME/app/venv/bin/pip install polars
RUN /home/$USERNAME/app/venv/bin/pip install pandas
RUN /home/$USERNAME/app/venv/bin/pip install pyarrow
RUN /home/$USERNAME/app/venv/bin/pip install matplotlib
RUN /home/$USERNAME/app/venv/bin/pip install seaborn
RUN /home/$USERNAME/app/venv/bin/pip install jupyterlab
RUN /home/$USERNAME/app/venv/bin/pip install notebook
RUN /home/$USERNAME/app/venv/bin/pip install ipykernel

USER $USERNAME
ENV PATH=/home/$USERNAME/app/venv/bin:$PATH
ENV PATH=/workspace/feature_extraction/target/release:$PATH

# Set environment variables for Rust
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
# Activate virtual environment on login
RUN echo 'source /home/$USERNAME/app/venv/bin/activate' >> $HOME/.bashrc
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/workspace"' >> $HOME/.bashrc