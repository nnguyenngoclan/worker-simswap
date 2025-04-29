FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip unzip wget build-essential cmake libboost-all-dev libgtk-3-dev pkg-config git

RUN ldconfig /usr/local/cuda-12.1/compat/

RUN pip install --upgrade pip setuptools wheel

RUN pip install dlib

# Install Python dependencies (Worker Template)
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

ADD src .

RUN wget -O arcfacemodel.zip "https://drive.usercontent.google.com/download?id=1n3MKPOiaAPy8YX9UvXjtPjTXKDMO_62I&authuser=0&confirm=t&uuid=3336e637-553e-46c0-a6d2-ea7373ef9403&at=APcmpox937rsM9pOHqXMgwhwLp7_%3A1745573263774" \
    && unzip -o arcfacemodel.zip -d / \
    && rm arcfacemodel.zip

RUN wget -O checkpoints.zip "https://drive.usercontent.google.com/download?id=1WGAJWNAjowKWCsaaaduYNWF4wg6LSNFY&export=download&authuser=0&confirm=t&uuid=39d2e05a-148f-43b3-86cb-2e61b43e6325&at=APcmpoy1OHijleOpguOhxU4dzu7C%3A1745575580882" \
    && unzip -o checkpoints.zip -d / \
    && rm checkpoints.zip

RUN mkdir -p /insightface_func/models

RUN wget -O facemodel.zip "https://drive.usercontent.google.com/download?id=1wzabOH2Tln_OF5NwCPrhcCgwf31KnOIs&export=download&authuser=0&confirm=t&uuid=7e9895a7-b429-440d-9b36-6482364287d7&at=APcmpoyl96L4gSGzy8kX_ZMzJb47%3A1745574985501" \
    && unzip -o facemodel.zip -d /insightface_func/models \
    && rm facemodel.zip

RUN wget -O checkpoint.zip "https://drive.usercontent.google.com/download?id=1mmNs_90eFC-EJaBRN6SwWeIEJKxTcGzs&export=download&authuser=0&confirm=t&uuid=ba383e93-4568-49a5-b242-3d008c235515&at=APcmpowWLiWxJesgW4Wmoa3r5PwI%3A1745634383056" \
    && unzip -o checkpoint.zip -d /parsing_model \
    && rm checkpoint.zip    

RUN wget -O upload_file_key.json "https://drive.google.com/uc?export=download&id=16OWrYM4ubCmT2sZZJxSgdTfVfk9NJiPz" 

# Add src files (Worker Template)
ENV RUNPOD_DEBUG_LEVEL=INFO

CMD python3 -u /rp_handler.py
