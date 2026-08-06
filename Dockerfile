# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for audio processing (Whisper/Edge-TTS)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install PyTorch built for CUDA 12.1 first so it matches the NVIDIA driver on
# the host (e.g. Tesla P100, whose driver supports up to CUDA 12.2). The default
# torch wheel on PyPI targets a newer CUDA runtime and fails to initialize on
# this driver, silently falling back to CPU with a "driver is too old" warning.
# The cu121 wheels bundle their own CUDA runtime and include Pascal (sm_60)
# support, so only the host driver + nvidia-container-toolkit are needed.
# Installing torch here means the sentence-transformers install below sees it as
# already satisfied and won't replace it with the mismatched wheel.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# faster-whisper (CTranslate2) needs cuDNN/cuBLAS at runtime to use the GPU.
# The cu121 torch wheel already ships these as nvidia-* packages, so point the
# dynamic linker at them instead of installing CUDA libs a second time.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib

# Copy your requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose port 8000 so the frontend can reach FastAPI
EXPOSE 8000

# Start the FastAPI server when the container boots
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]