"""
Скрипт для анализа результатов бенчмарков NVIDIA GB10
"""
import json
import glob
import os
from pathlib import Path
from collections import defaultdict

def analyze_benchmarks():
    """Анализирует все файлы бенчмарков и выводит ключевые метрики"""

    # Собираем все summary файлы
    summary_files = glob.glob("final_results/**/benchmark_summary*.json", recursive=True)

    results = defaultdict(lambda: defaultdict(list))

    for filepath in sorted(summary_files):
        with open(filepath, 'r') as fp:
            data = json.load(fp)

        # Извлекаем название модели из имени файла
        filename = os.path.basename(filepath)
        # Формат: benchmark_summary_Qwen_Qwen3-32B-AWQ_20251221_172513.json
        parts = filename.replace("benchmark_summary_", "").split("_2025")[0]
        model_name = parts

        for entry in data:
            dataset = entry.get('dataset', 'unknown')
            conc = entry.get('max_concurrency', 0)
            result = entry.get('result', {})

            metrics = {
                'file': filepath,
                'model': model_name,
                'dataset': dataset,
                'concurrency': conc,
                'output_throughput': result.get('output_throughput', 0),
                'total_token_throughput': result.get('total_token_throughput', 0),
                'mean_ttft_ms': result.get('mean_ttft_ms', 0),
                'median_ttft_ms': result.get('median_ttft_ms', 0),
                'p99_ttft_ms': result.get('p99_ttft_ms', 0),
                'mean_tpot_ms': result.get('mean_tpot_ms', 0),
                'median_tpot_ms': result.get('median_tpot_ms', 0),
                'p99_tpot_ms': result.get('p99_tpot_ms', 0),
                'total_input_tokens': result.get('total_input_tokens', 0),
                'total_output_tokens': result.get('total_output_tokens', 0),
                'max_concurrent_requests': result.get('max_concurrent_requests', 0),
                'completed': result.get('completed', 0),
                'num_prompts': result.get('num_prompts', 0),
                'duration': result.get('duration', 0),
                'request_throughput': result.get('request_throughput', 0),
            }

            results[model_name][dataset].append(metrics)

    # Выводим результаты
    print("=" * 100)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ БЕНЧМАРКОВ NVIDIA GB10")
    print("=" * 100)

    for model in sorted(results.keys()):
        print(f"\n{'='*90}")
        print(f"МОДЕЛЬ: {model}")
        print(f"{'='*90}")

        datasets = results[model]

        for dataset in sorted(datasets.keys()):
            runs = datasets[dataset]
            runs_sorted = sorted(runs, key=lambda x: x['concurrency'])

            print(f"\n  Dataset: {dataset}")
            print(f"  {'-'*88}")
            print(f"  {'Conc':<6} {'Out TP':<10} {'Tot TP':<10} {'TTFT':<8} {'TPOT':<8} "
                  f"{'MaxConc':<8} {'Done':<6} {'Req/s':<8}")
            print(f"  {'-'*88}")

            for r in runs_sorted:
                print(f"  {r['concurrency']:<6} "
                      f"{r['output_throughput']:>9.1f} "
                      f"{r['total_token_throughput']:>9.1f} "
                      f"{r['mean_ttft_ms']:>7.1f} "
                      f"{r['mean_tpot_ms']:>7.1f} "
                      f"{r['max_concurrent_requests']:>7} "
                      f"{r['completed']:>5} "
                      f"{r['request_throughput']:>7.2f}")

            # Находим точку насыщения
            max_tp_idx = max(range(len(runs_sorted)),
                           key=lambda i: runs_sorted[i]['output_throughput'])
            max_tp = runs_sorted[max_tp_idx]

            print(f"\n  >> Пик throughput: {max_tp['output_throughput']:.1f} tok/s "
                  f"при concurrency={max_tp['concurrency']}")

            # Анализ плато
            if max_tp_idx < len(runs_sorted) - 1:
                plateau_runs = runs_sorted[max_tp_idx:]
                avg_plateau_tp = sum(r['output_throughput'] for r in plateau_runs) / len(plateau_runs)
                variance = max(r['output_throughput'] for r in plateau_runs) - min(r['output_throughput'] for r in plateau_runs)
                print(f"  >> Плато начинается при concurrency={max_tp['concurrency']}")
                print(f"  >> Средний throughput на плато: {avg_plateau_tp:.1f} tok/s")
                print(f"  >> Вариация на плато: {variance:.1f} tok/s ({variance/avg_plateau_tp*100:.1f}%)")

                # Анализ TTFT на плато
                avg_ttft_plateau = sum(r['mean_ttft_ms'] for r in plateau_runs) / len(plateau_runs)
                print(f"  >> Средний TTFT на плато: {avg_ttft_plateau:.1f} ms")

            print()

    # Сводная таблица по моделям
    print("\n" + "="*90)
    print("СВОДНАЯ ТАБЛИЦА: МАКСИМАЛЬНАЯ ПРОИЗВОДИТЕЛЬНОСТЬ")
    print("="*90)
    print(f"{'Модель':<40} {'Dataset':<25} {'Max TP':<12} {'@Conc':<8}")
    print("-"*90)

    summary_data = []
    for model in sorted(results.keys()):
        for dataset in sorted(results[model].keys()):
            runs = results[model][dataset]
            max_run = max(runs, key=lambda x: x['output_throughput'])
            summary_data.append({
                'model': model,
                'dataset': dataset,
                'max_tp': max_run['output_throughput'],
                'concurrency': max_run['concurrency'],
                'ttft': max_run['mean_ttft_ms'],
                'tpot': max_run['mean_tpot_ms']
            })
            print(f"{model:<40} {dataset:<25} {max_run['output_throughput']:>10.1f} "
                  f"{max_run['concurrency']:>7}")

    print("\n" + "="*90)
    print("АНАЛИЗ ПО РАЗМЕРУ МОДЕЛЕЙ")
    print("="*90)

    # Группируем по размеру модели
    model_sizes = defaultdict(list)
    for item in summary_data:
        if '0.6B' in item['model']:
            size = '0.6B'
        elif '4B' in item['model']:
            size = '4B'
        elif '32B' in item['model']:
            size = '32B'
        else:
            size = 'unknown'
        model_sizes[size].append(item)

    for size in sorted(model_sizes.keys()):
        items = model_sizes[size]
        print(f"\n{size}:")
        for item in items:
            quant = 'BF16'
            if 'FP8' in item['model']:
                quant = 'FP8'
            elif 'AWQ' in item['model']:
                quant = 'AWQ (4-bit)'
            elif 'gguf' in item['model'] or 'Q4' in item['model']:
                quant = 'GGUF Q4'

            print(f"  {quant:<15} {item['dataset']:<25} "
                  f"TP: {item['max_tp']:>8.1f} tok/s, "
                  f"TTFT: {item['ttft']:>6.1f}ms, "
                  f"TPOT: {item['tpot']:>6.1f}ms")

    return results, summary_data

if __name__ == "__main__":
    results, summary = analyze_benchmarks()
