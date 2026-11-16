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

# Build official OpenFace FeatureExtraction. Fetch the large CEN patch experts
# from a pinned mirror and verify their exact SHA-256 digests: the upstream
# download_models.sh accepts any HTTP 200 response, including an HTML error page.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git OpenFace \
    && cd /opt/OpenFace/lib/local/LandmarkDetector/model/patch_experts \
    && bash -ec ' \
         fetch_cen_model() { \
           file="$1"; expected_size="$2"; expected_sha="$3"; \
           url="https://huggingface.co/kabachuha/openface_cen_patches/resolve/25440f074a1d0e2473e450971178be9340083dd2/$file"; \
           wget -q --https-only --max-redirect=20 --timeout=60 --tries=3 -O "$file" "$url"; \
           actual_size="$(stat -c "%s" "$file")"; \
           test "$actual_size" = "$expected_size"; \
           printf "%s  %s\\n" "$expected_sha" "$file" | sha256sum -c -; \
         }; \
         fetch_cen_model cen_patches_0.25_of.dat 60602360 99d3df9888115428075de7b8f1bc86176881de640c2fdf47e2ab0cee556661bf; \
         fetch_cen_model cen_patches_0.35_of.dat 60602360 1070bff51b077ee6a18ce8e1ebbc6f568e1a7a469f6911a8263413878b3df469; \
         fetch_cen_model cen_patches_0.50_of.dat 154289792 3f08124067e326e83e3261a55c79493e745b01138c9c1b59713dd9b17424168e; \
         fetch_cen_model cen_patches_1.00_of.dat 154289792 bed961bbfa2cc41a709d44c1d1d8a92b2537bc46f2fd6ab430c137fedd0e21b0 \
       ' \
    && cd /opt/OpenFace \
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
