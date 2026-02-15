# Базовый образ NVIDIA с vLLM
FROM nvcr.io/nvidia/vllm:25.11-py3

# Метаданные
LABEL maintainer="aandreevich@palatine.ru"
LABEL description="LLM GPU Benchmark для GX10"
LABEL version="1.0"

# Установка переменных окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf_cache
ENV TORCH_HOME=/workspace/torch_cache
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    nvtop \
    wget \
    curl \
    mc \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей для визуализации
RUN pip install --no-cache-dir \
    plotly>=5.18.0 \
    pandas>=2.0.0 \
    kaleido>=0.2.1 \
    numpy>=1.24.0 \
    openai \
    jupyter

# Копирование датасета
COPY prepare_dataset/dataset /workspace/dataset/

# Копирование скриптов и конфигурации
COPY benchmark.py /workspace/benchmark.py
COPY visualize.py /workspace/visualize.py
COPY benchmark_config.json /workspace/benchmark_config.json

# Создание директорий для результатов и графиков
RUN mkdir -p /workspace/results /workspace/plots

WORKDIR /workspace

# Порт для vLLM сервера
EXPOSE 8000

