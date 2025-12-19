FROM ubuntu:latest

RUN apt update && apt install -y git wget
RUN apt install -y build-essential

RUN git clone https://github.com/Verified-Intelligence/auto_LiRPA.git
# RUN git clone https://github.com/ufukguenes/ICNN_verification.git

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh

ENV PATH="/root/miniconda3/bin:$PATH"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n icnn_env python=3.11

RUN conda run -n icnn_env pip install gurobipy
RUN conda run -n icnn_env pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN conda run -n icnn_env pip install matplotlib numpy ipykernel scipy

#RUN pip install wheel
RUN cd auto_LiRPA && conda run -n icnn_env pip install .
RUN conda run -n icnn_env pip install onnx vnnlib git+https://github.com/KaidiXu/onnx2pytorch jupyter nbconvert 
RUN conda run -n icnn_env pip install --upgrade pip ipython ipykernel
RUN conda run -n icnn_env python -m ipykernel install --user --name=icnn_env

CMD ["bash", "-c", "conda activate icnn_env"]