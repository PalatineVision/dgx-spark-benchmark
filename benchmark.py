#!/usr/bin/env python3
"""
Скрипт для запуска бенчмарков vLLM на списке моделей с различными параметрами.

Запускает vllm bench serve для каждой комбинации модели и параметров,
сохраняя результаты в уникальные JSON файлы.
"""

import subprocess
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class ModelConfig:
    """
    Конфигурация модели с указанием хоста, порта и метаданных.

    Attributes:
        name: Имя модели (например, "Qwen/Qwen3-0.6B")
        port: Порт, на котором запущен vLLM сервер для этой модели
        metadata: Строка метаданных для включения в имя файла результатов
        host: Хост сервера (например, "http://vllm-server"), если пусто - используется base_host
    """
    name: str
    port: int
    metadata: str = ""
    host: str = ""


@dataclass
class BenchmarkConfig:
    """
    Конфигурация для запуска бенчмарка.

    Attributes:
        models: Список моделей для тестирования (с указанием портов)
        max_concurrency_values: Список значений max_concurrency для тестирования
        num_prompts: Количество промптов для каждого теста
        input_lengths: Список длин входных токенов
        output_lengths: Список длин выходных токенов
        results_dir: Директория для сохранения результатов
        dataset_name: Название датасета (random, spec_bench и т.д.)
        base_host: Базовый хост vLLM серверов (без порта)
    """
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig("Qwen/Qwen3-0.6B", 8000, "")])
    max_concurrency_values: list[int] = field(default_factory=lambda: [1, 10, 50, 100])
    num_prompts: int = 10
    input_lengths: list[int] = field(default_factory=lambda: [128, 512, 1024, 3000])
    output_lengths: list[int] = field(default_factory=lambda: [128, 512, 1024])
    results_dir: str = "/workspace/results"
    dataset_name: str = "random"
    dataset_path: list[str] = field(default_factory=lambda: ["/workspace/dataset/chat_dataset.jsonl", "/workspace/dataset/rag_dataset.jsonl"])
    base_host: str = "http://localhost"
    timeout: int = 600  # Таймаут на один тест в секундах

    def get_base_url(self, model: ModelConfig) -> str:
        """
        Формирует полный URL для модели.

        Использует host из конфигурации модели, если указан,
        иначе использует глобальный base_host.

        Args:
            model: Конфигурация модели

        Returns:
            Полный URL вида http://host:port
        """
        host = model.host if model.host else self.base_host
        return f"{host}:{model.port}"


def sanitize_model_name(model: str) -> str:
    """
    Преобразует имя модели в безопасное имя файла.

    Args:
        model: Имя модели (например, "Qwen/Qwen3-0.6B")

    Returns:
        Безопасное имя для файла (например, "Qwen_Qwen3-0.6B")
    """
    return model.replace("/", "_").replace(":", "_")


def get_dataset_name_from_path(dataset_path: str) -> str:
    """
    Извлекает имя датасета из пути к файлу (без расширения).

    Args:
        dataset_path: Путь к файлу датасета (например, "/workspace/dataset/chat_dataset.jsonl")

    Returns:
        Имя файла без расширения (например, "chat_dataset")
    """
    return Path(dataset_path).stem


def generate_result_filename(
    model: str,
    max_concurrency: int,
    input_len: int,
    output_len: int,
    dataset_name: str,
    dataset_file: Optional[str] = None
) -> str:
    """
    Генерирует уникальное имя файла для результатов бенчмарка.

    Args:
        model: Имя модели
        max_concurrency: Значение max_concurrency
        input_len: Длина входных токенов
        output_len: Длина выходных токенов
        dataset_name: Название датасета (random, custom, spec_bench и т.д.)
        dataset_file: Имя файла датасета (без расширения), если используется custom

    Returns:
        Уникальное имя файла с временной меткой
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = sanitize_model_name(model)

    # Если указан файл датасета, используем его имя вместо dataset_name
    ds_label = dataset_file if dataset_file else dataset_name

    return f"{safe_model}_conc{max_concurrency}_in{input_len}_out{output_len}_{ds_label}_{timestamp}.json"


def wait_for_server(base_url: str, max_retries: int = 60, delay: int = 10) -> bool:
    """
    Ожидает готовности vLLM сервера.

    Args:
        base_url: URL сервера
        max_retries: Максимальное количество попыток
        delay: Задержка между попытками в секундах

    Returns:
        True если сервер готов, False в противном случае
    """
    import urllib.request
    import urllib.error

    health_url = f"{base_url}/health"

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    print(f"[INFO] vLLM сервер готов к работе")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[INFO] Ожидание сервера... попытка {attempt + 1}/{max_retries}")
            time.sleep(delay)

    print(f"[ERROR] Сервер не отвечает после {max_retries} попыток")
    return False


def run_benchmark(
    model: str,
    max_concurrency: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    result_filename: str,
    results_dir: str,
    dataset_name: str = "random",
    dataset_path: Optional[str] = None,
    base_url: str = "http://vllm-server:8000",
    timeout: int = 600
) -> Optional[dict]:
    """
    Запускает один бенчмарк с заданными параметрами.

    Args:
        model: Имя модели для бенчмарка
        max_concurrency: Максимальное количество параллельных запросов
        num_prompts: Количество промптов
        input_len: Длина входных токенов
        output_len: Длина выходных токенов
        result_filename: Имя файла для сохранения результатов
        results_dir: Директория для результатов
        dataset_name: Название датасета (random, custom, spec_bench и т.д.)
        dataset_path: Путь к файлу датасета (для custom датасета)
        base_url: URL vLLM сервера
        timeout: Таймаут выполнения в секундах

    Returns:
        Словарь с результатами или None в случае ошибки
    """
    result_path = os.path.join(results_dir, result_filename)

    cmd = [
        "vllm", "bench", "serve",
        "--model", model,
        "--base-url", base_url,
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--dataset-name", dataset_name,
        "--ignore-eos",
        "--max-concurrency", str(max_concurrency),
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--trust-remote-code",
        "--save-result",
        "--result-filename", result_path,
    ]

    # Добавляем путь к датасету для custom датасетов
    if dataset_name == "custom" and dataset_path:
        cmd.extend(["--dataset-path", dataset_path])

    dataset_file = get_dataset_name_from_path(dataset_path) if dataset_path else None

    print(f"\n{'='*60}")
    print(f"[BENCHMARK] Запуск теста:")
    print(f"  Модель: {model}")
    print(f"  Concurrency: {max_concurrency}")
    print(f"  Input length: {input_len}")
    print(f"  Output length: {output_len}")
    if dataset_path:
        print(f"  Датасет: {dataset_file} ({dataset_path})")
    else:
        print(f"  Датасет: {dataset_name}")
    print(f"  Результат: {result_filename}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"[ERROR] Бенчмарк завершился с ошибкой:")
            print(result.stderr)
            return None

        # Читаем результаты из файла
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                return json.load(f)
        else:
            print(f"[WARNING] Файл результатов не найден: {result_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Таймаут выполнения бенчмарка ({timeout}s)")
        return None
    except Exception as e:
        print(f"[ERROR] Ошибка при выполнении бенчмарка: {e}")
        return None


def run_all_benchmarks(config: BenchmarkConfig) -> dict[str, list[dict]]:
    """
    Запускает все бенчмарки согласно конфигурации.

    Итерируется по всем комбинациям: модели, датасеты, concurrency, input/output lengths.
    Для каждой модели создаётся отдельный summary файл.

    Args:
        config: Конфигурация бенчмарков

    Returns:
        Словарь с результатами по моделям {model_name: [results]}
    """
    # Создаём директорию для результатов
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Определяем список датасетов для итерации
    # Если dataset_name == "custom" и есть dataset_path, итерируемся по путям
    # Иначе используем один раз с dataset_name (random, spec_bench и т.д.)
    if config.dataset_name == "custom" and config.dataset_path:
        datasets_to_run = config.dataset_path
    else:
        datasets_to_run = [None]  # None означает использование только dataset_name

    num_datasets = len(datasets_to_run)
    total_tests = (
        len(config.models) *
        num_datasets *
        len(config.max_concurrency_values) *
        len(config.input_lengths) *
        len(config.output_lengths)
    )
    current_test = 0

    print(f"\n[INFO] Всего запланировано тестов: {total_tests}")
    print(f"[INFO] Модели: {[m.name for m in config.models]}")
    if config.dataset_name == "custom":
        print(f"[INFO] Датасеты: {[get_dataset_name_from_path(d) for d in datasets_to_run if d]}")

    for model_config in config.models:
        model_name = model_config.name
        base_url = config.get_base_url(model_config)

        print(f"\n{'='*60}")
        print(f"[MODEL] Тестирование модели: {model_name}")
        print(f"[MODEL] URL сервера: {base_url}")
        print(f"{'='*60}")

        # Ожидаем готовности сервера для этой модели
        if not wait_for_server(base_url):
            print(f"[ERROR] Невозможно подключиться к серверу для модели {model_name}")
            print(f"[WARNING] Пропуск модели {model_name}")
            continue

        model_results = []

        for dataset_path in datasets_to_run:
            # Получаем имя файла датасета для использования в имени результата
            dataset_file = get_dataset_name_from_path(dataset_path) if dataset_path else None

            for max_concurrency in config.max_concurrency_values:
                for input_len in config.input_lengths:
                    for output_len in config.output_lengths:
                        current_test += 1
                        print(f"\n[PROGRESS] Тест {current_test}/{total_tests}")

                        result_filename = generate_result_filename(
                            model=model_name,
                            max_concurrency=max_concurrency,
                            input_len=input_len,
                            output_len=output_len,
                            dataset_name=config.dataset_name,
                            dataset_file=dataset_file
                        )

                        result = run_benchmark(
                            model=model_name,
                            max_concurrency=max_concurrency,
                            num_prompts=config.num_prompts,
                            input_len=input_len,
                            output_len=output_len,
                            result_filename=result_filename,
                            results_dir=config.results_dir,
                            dataset_name=config.dataset_name,
                            dataset_path=dataset_path,
                            base_url=base_url,
                            timeout=config.timeout
                        )

                        if result:
                            model_results.append({
                                "model": model_name,
                                "dataset": dataset_file if dataset_file else config.dataset_name,
                                "dataset_path": dataset_path,
                                "max_concurrency": max_concurrency,
                                "input_len": input_len,
                                "output_len": output_len,
                                "filename": result_filename,
                                "result": result
                            })

                        # Небольшая пауза между тестами
                        time.sleep(2)

        # Сохраняем сводный отчёт для этой модели
        safe_model_name = sanitize_model_name(model_name)
        # Добавляем metadata в имя файла, если она указана
        metadata_suffix = f"_{model_config.metadata}" if model_config.metadata else ""
        summary_path = os.path.join(
            config.results_dir,
            f"benchmark_summary_{safe_model_name}{metadata_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] Сводный отчёт для {model_name} сохранён: {summary_path}")
        print(f"[INFO] Завершено тестов для модели: {len(model_results)}")

        all_results[model_name] = model_results

    print(f"\n[INFO] Всего завершено тестов: {sum(len(r) for r in all_results.values())}/{total_tests}")

    return all_results


def parse_model_string(model_str: str) -> ModelConfig:
    """
    Парсит строку модели в формате "model@host:port:metadata" или "model:port:metadata".

    Формат с хостом: "Qwen/Qwen3-0.6B@http://vllm-server:8000:fp16"
    Формат без хоста: "Qwen/Qwen3-0.6B:8000:fp16" (используется base_host)

    Args:
        model_str: Строка с конфигурацией модели

    Returns:
        ModelConfig с именем, портом, метаданными и хостом
    """
    model_str = model_str.strip()
    host = ""

    # Проверяем наличие хоста (разделитель @)
    if "@" in model_str:
        name_part, rest = model_str.split("@", 1)
        name = name_part
        # rest содержит host:port:metadata, например "http://vllm-server:8000:fp16"
        # Нужно найти порт после хоста
        # Формат: scheme://hostname:port:metadata
        if "://" in rest:
            scheme_and_rest = rest.split("://", 1)
            scheme = scheme_and_rest[0]
            host_port_meta = scheme_and_rest[1].split(":")
            hostname = host_port_meta[0]
            host = f"{scheme}://{hostname}"
            port = int(host_port_meta[1]) if len(host_port_meta) > 1 else 8000
            metadata = ":".join(host_port_meta[2:]) if len(host_port_meta) > 2 else ""
        else:
            # Формат без схемы: hostname:port:metadata
            parts = rest.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 8000
            metadata = ":".join(parts[2:]) if len(parts) > 2 else ""
    else:
        # Формат без хоста: model:port:metadata
        parts = model_str.split(":")
        name = parts[0]
        port = 8000
        metadata = ""

        if len(parts) >= 2 and parts[1].isdigit():
            port = int(parts[1])
            if len(parts) >= 3:
                metadata = ":".join(parts[2:])
        elif len(parts) >= 2:
            metadata = ":".join(parts[1:])

    return ModelConfig(name=name, port=port, metadata=metadata, host=host)


def load_config_from_env() -> BenchmarkConfig:
    """
    Загружает конфигурацию из переменных окружения.

    Формат моделей: "model1:port1,model2:port2" или "model1,model2" (порт по умолчанию 8000)

    Returns:
        Конфигурация бенчмарка
    """
    config = BenchmarkConfig()

    # Загрузка моделей (через запятую, формат: model:port)
    if os.getenv("BENCHMARK_MODELS"):
        config.models = [
            parse_model_string(m) for m in os.getenv("BENCHMARK_MODELS").split(",")
        ]

    # Загрузка значений concurrency (через запятую)
    if os.getenv("BENCHMARK_CONCURRENCY"):
        config.max_concurrency_values = [
            int(c.strip()) for c in os.getenv("BENCHMARK_CONCURRENCY").split(",")
        ]

    # Количество промптов
    if os.getenv("BENCHMARK_NUM_PROMPTS"):
        config.num_prompts = int(os.getenv("BENCHMARK_NUM_PROMPTS"))

    # Длины входных токенов (через запятую)
    if os.getenv("BENCHMARK_INPUT_LENGTHS"):
        config.input_lengths = [
            int(l.strip()) for l in os.getenv("BENCHMARK_INPUT_LENGTHS").split(",")
        ]

    # Длины выходных токенов (через запятую)
    if os.getenv("BENCHMARK_OUTPUT_LENGTHS"):
        config.output_lengths = [
            int(l.strip()) for l in os.getenv("BENCHMARK_OUTPUT_LENGTHS").split(",")
        ]

    # Директория результатов
    if os.getenv("BENCHMARK_RESULTS_DIR"):
        config.results_dir = os.getenv("BENCHMARK_RESULTS_DIR")

    # Датасет
    if os.getenv("BENCHMARK_DATASET"):
        config.dataset_name = os.getenv("BENCHMARK_DATASET")

    # Пути к датасетам (через запятую)
    if os.getenv("BENCHMARK_DATASET_PATH"):
        config.dataset_path = [
            p.strip() for p in os.getenv("BENCHMARK_DATASET_PATH").split(",")
        ]

    # Базовый хост серверов
    if os.getenv("BENCHMARK_BASE_HOST"):
        config.base_host = os.getenv("BENCHMARK_BASE_HOST")

    # Таймаут
    if os.getenv("BENCHMARK_TIMEOUT"):
        config.timeout = int(os.getenv("BENCHMARK_TIMEOUT"))

    return config


def load_config_from_file(config_path: str) -> BenchmarkConfig:
    """
    Загружает конфигурацию из JSON файла.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Конфигурация бенчмарка
    """
    with open(config_path, 'r') as f:
        data = json.load(f)

    # Парсим модели из нового формата [{name, port, metadata, host}, ...]
    models_data = data.get("models", [{"name": "Qwen/Qwen3-0.6B", "port": 8000}])
    models = [
        ModelConfig(
            name=m["name"],
            port=m["port"],
            metadata=m.get("metadata", ""),
            host=m.get("host", "")
        )
        for m in models_data
    ]

    return BenchmarkConfig(
        models=models,
        max_concurrency_values=data.get("max_concurrency_values", [1, 10, 50, 100]),
        num_prompts=data.get("num_prompts", 10),
        input_lengths=data.get("input_lengths", [128, 512, 1024, 3000]),
        output_lengths=data.get("output_lengths", [128, 512, 1024]),
        results_dir=data.get("results_dir", "/workspace/results"),
        dataset_name=data.get("dataset_name", "random"),
        dataset_path=data.get("dataset_path", ["/workspace/dataset/chat_dataset.jsonl", "/workspace/dataset/rag_dataset.jsonl"]),
        base_host=data.get("base_host", "http://localhost"),
        timeout=data.get("timeout", 600)
    )


def main():
    """
    Главная функция запуска бенчмарков.
    """
    print("="*60)
    print("  vLLM Benchmark Runner")
    print("="*60)

    # Загружаем конфигурацию
    config_path = os.getenv("BENCHMARK_CONFIG_FILE")

    if config_path and os.path.exists(config_path):
        print(f"[INFO] Загрузка конфигурации из файла: {config_path}")
        config = load_config_from_file(config_path)
    else:
        print("[INFO] Загрузка конфигурации из переменных окружения")
        config = load_config_from_env()

    print(f"\n[CONFIG] Модели:")
    for m in config.models:
        host_info = f", host: {m.host}" if m.host else ""
        metadata_info = f", metadata: {m.metadata}" if m.metadata else ""
        print(f"         - {m.name} (порт: {m.port}{host_info}{metadata_info})")
    print(f"[CONFIG] Concurrency: {config.max_concurrency_values}")
    print(f"[CONFIG] Num prompts: {config.num_prompts}")
    print(f"[CONFIG] Input lengths: {config.input_lengths}")
    print(f"[CONFIG] Output lengths: {config.output_lengths}")
    print(f"[CONFIG] Dataset name: {config.dataset_name}")
    if config.dataset_path:
        print(f"[CONFIG] Dataset paths: {config.dataset_path}")
    print(f"[CONFIG] Results dir: {config.results_dir}")
    print(f"[CONFIG] Base host: {config.base_host}")

    # Запускаем бенчмарки
    results = run_all_benchmarks(config)

    print("\n" + "="*60)
    print("  Бенчмарки завершены!")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
