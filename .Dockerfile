# Use official PyTorch CUDA image (Python 3.11 + CUDA 12.6)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for some packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py README.md /app/
COPY src /app/src

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA (specific versions)
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
RUN pip install --no-cache-dir -e .

RUN rm -rf /app/src

# Expose port if needed (e.g., tensorboard)
EXPOSE 6006

ENTRYPOINT ["tail", "-f", "/dev/null"]
