ARG SRC_IMAGE
FROM ${SRC_IMAGE}

SHELL ["/bin/bash", "-c"]
RUN sed -i.bak -e "s%http://[^ ]\+%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-numpy-dev \
        libpcl-dev \
        libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_numpy3*.so /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ; \
    fi; \
    if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_python3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_python3*.so /usr/lib/x86_64-linux-gnu/libboost_python3.so ; \
    fi

RUN if [[ -f /opt/conda/bin/activate ]]; then \
        source /opt/conda/bin/activate && \
        conda install -c conda-forge scikit-build ninja h5py==2.10.0 argcomplete optuna optuna-dashboard openpyxl pymysql; \
    else \
        pip3 install scikit-build ninja h5py==2.10.0 argcomplete optuna optuna-dashboard openpyxl pymysql; \
    fi; \
    pip install --verbose git+https://github.com/shikishima-TasakiLab/pointsmap-python && \
    pip install git+https://github.com/shikishima-TasakiLab/h5dataloader && \
    pip install git+https://github.com/shikishima-TasakiLab/h5datacreator && \
    pip install adabelief-pytorch

RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof tensorboard && \
    pip install tensorboard torch-tb-profiler==0.1.0
