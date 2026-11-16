# NeuroSense backend with the OpenFace FeatureExtraction binary.
# OpenFace is required because the AFFEC model expects OpenFace AU/gaze/pose inputs.
FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OPENFACE_ROOT=/opt/OpenFace \
    OPENFACE_FEATURE_EXTRACTION_PATH=/opt/OpenFace/build/bin/FeatureExtraction \
    NEUROSENSE_FACE_VALENCE_MODEL_PATH=/app/ml_model/face_valence/neurosense_face_valence_rf.joblib

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config wget ca-certificates ffmpeg \
    libopencv-dev libopenblas-dev liblapack-dev libdlib-dev \
    libboost-filesystem-dev libboost-system-dev libtbb-dev \
    libavcodec-dev libavformat-dev libswscale-dev libgtk-3-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Build official OpenFace FeatureExtraction and download its model files.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git OpenFace \
    && cd /opt/OpenFace \
    && bash download_models.sh \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
    && cmake --build build --target FeatureExtraction -j2 \
    && test -x /opt/OpenFace/build/bin/FeatureExtraction \
    && ldconfig

WORKDIR /app
COPY requirements-render.txt ./requirements-render.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-render.txt

COPY . .

EXPOSE 10000
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
