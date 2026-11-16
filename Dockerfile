# NeuroSense backend with a checksum-verified OpenFace FeatureExtraction binary.
# The AFFEC felt-valence pipeline expects OpenFace AU/gaze/pose columns.
FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OPENFACE_ROOT=/opt/OpenFace \
    OPENFACE_FEATURE_EXTRACTION_PATH=/opt/OpenFace/build/bin/FeatureExtraction \
    NEUROSENSE_FACE_VALENCE_MODEL_PATH=/app/ml_model/face_valence/neurosense_face_valence_rf.joblib \
    NEUROSENSE_AFFEC_PYTHON=/opt/affec-venv/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config wget ca-certificates ffmpeg \
    libopencv-dev libopenblas-dev liblapack-dev libdlib-dev \
    libboost-filesystem-dev libboost-system-dev libtbb-dev \
    libavcodec-dev libavformat-dev libswscale-dev libgtk-3-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Build OpenFace from source. Its CE-CLM detector requires all four CEN
# patch-expert binaries. The upstream downloader uses legacy file hosts and
# only checks whether a file exists; download pinned, checksum-verified
# copies instead so an HTML/error page cannot pass the image build.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git OpenFace \
    && cd /opt/OpenFace \
    && patch_dir=/opt/OpenFace/lib/local/LandmarkDetector/model/patch_experts \
    && mkdir -p "$patch_dir" \
    && cd "$patch_dir" \
    && wget --https-only --max-redirect=20 --tries=3 --timeout=90 -O cen_patches_0.25_of.dat https://huggingface.co/kabachuha/openface_cen_patches/resolve/25440f074a1d0e2473e450971178be9340083dd2/cen_patches_0.25_of.dat \
    && wget --https-only --max-redirect=20 --tries=3 --timeout=90 -O cen_patches_0.35_of.dat https://huggingface.co/kabachuha/openface_cen_patches/resolve/25440f074a1d0e2473e450971178be9340083dd2/cen_patches_0.35_of.dat \
    && wget --https-only --max-redirect=20 --tries=3 --timeout=90 -O cen_patches_0.50_of.dat https://huggingface.co/kabachuha/openface_cen_patches/resolve/25440f074a1d0e2473e450971178be9340083dd2/cen_patches_0.50_of.dat \
    && wget --https-only --max-redirect=20 --tries=3 --timeout=90 -O cen_patches_1.00_of.dat https://huggingface.co/kabachuha/openface_cen_patches/resolve/25440f074a1d0e2473e450971178be9340083dd2/cen_patches_1.00_of.dat \
    && echo "99d3df9888115428075de7b8f1bc86176881de640c2fdf47e2ab0cee556661bf  cen_patches_0.25_of.dat" | sha256sum -c - \
    && echo "1070bff51b077ee6a18ce8e1ebbc6f568e1a7a469f6911a8263413878b3df469  cen_patches_0.35_of.dat" | sha256sum -c - \
    && echo "3f08124067e326e83e3261a55c79493e745b01138c9c1b59713dd9b17424168e  cen_patches_0.50_of.dat" | sha256sum -c - \
    && echo "bed961bbfa2cc41a709d44c1d1d8a92b2537bc46f2fd6ab430c137fedd0e21b0  cen_patches_1.00_of.dat" | sha256sum -c - \
    && test "$(stat -c%s cen_patches_0.25_of.dat)" -eq 60602360 \
    && test "$(stat -c%s cen_patches_0.35_of.dat)" -eq 60602360 \
    && test "$(stat -c%s cen_patches_0.50_of.dat)" -eq 154289792 \
    && test "$(stat -c%s cen_patches_1.00_of.dat)" -eq 154289792 \
    && cd /opt/OpenFace \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
    && cmake --build build --target FeatureExtraction -j2 \
    && test -x /opt/OpenFace/build/bin/FeatureExtraction \
    && ldconfig

WORKDIR /app
COPY requirements-render.txt ./requirements-render.txt
COPY requirements-affec-worker.txt ./requirements-affec-worker.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-render.txt \
    && python -m venv /opt/affec-venv \
    && /opt/affec-venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/affec-venv/bin/python -m pip install --no-cache-dir -r requirements-affec-worker.txt

COPY . .

EXPOSE 10000
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
