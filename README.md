# 🚀 DeepSeek-OCR GPU-Enabled Development Environment

This repository provides a fully containerized development setup for running **DeepSeek-OCR** with **GPU acceleration**, using:

- **VS Code Dev Containers**
- **NVIDIA CUDA Docker images**
- **uv** for fast & reproducible Python environments
- **A clean project layout** for inference scripts and model code

The environment automatically configures Python, installs dependencies, and enables GPU support if available.

---

## Features

- **GPU support** via CUDA (Dockerfile.gpu)  
- **Automatic Python environment** using `uv venv`  
- **Editable install** for local development (`src/deepseek_ocr`)  
- **Example inference script** (`run_model_script.py`)  

---

## Running in VS Code (Recommended)

1. Open the project folder in **VS Code**  
2. Press **Ctrl + Shift + P**  
3. Choose **“Dev Containers: Rebuild and Reopen in Container”**  

VS Code will:

- Build the GPU-enabled Docker image  
- Create a `.venv` environment with `uv`  
- Install dependencies listed in `pyproject.toml`  
- Activate the environment automatically  

You're ready to develop!

---

## Manually Activate the Environment

If the virtual environment is **not automatically activated**, run:

```bash
source .venv/bin/activate
```

## Run the Inference Example

An example DeepSeek-OCR inference script lives in:

```bash
python run_model_script.py
```

## GPU Requirements

To use the GPU devcontainer, you must have:

- NVIDIA GPU

- NVIDIA drivers installed

- Docker + NVIDIA Container Toolkit

- CUDA-compatible GPU (your Dockerfile uses CUDA 11.8)

Test GPU availability inside the container:

```bash
nvidia-smi
```
