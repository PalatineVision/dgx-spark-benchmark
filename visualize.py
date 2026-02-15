#!/usr/bin/env python3
"""
Скрипт для визуализации результатов бенчмарков vLLM.

Загружает данные из benchmark_summary_*.json и строит графики с помощью plotly express.
Поддерживает визуализацию как отдельных моделей, так и сравнительные графики.
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Флаг для проверки возможности экспорта в PNG
CAN_EXPORT_PNG = True

# Профессиональная цветовая палитра с хорошей различимостью
CHART_COLORS = [
    '#2E86AB',  # Синий
    '#E94F37',  # Красный
    '#1B998B',  # Бирюзовый
    '#F39237',  # Оранжевый
    '#8338EC',  # Фиолетовый
    '#06D6A0',  # Зелёный
]

# Стили для графиков
CHART_STYLE = {
    'plot_bgcolor': '#FAFAFA',
    'paper_bgcolor': 'white',
    'grid_color': '#E8E8E8',
    'line_color': '#CCCCCC',
    'font_family': 'Arial',
}


def format_chart_title(
    base_title: str, suffix: str = ''
) -> str:
    """Форматирует заголовок графика с переносом строки
    для длинных названий."""
    if not suffix:
        return base_title
    title = (
        f'{base_title}<br>'
        f'<span style="font-size:14px">({suffix})</span>'
    )
    return title


def save_figure(fig, output_dir: str, name: str) -> None:
    """
    Сохраняет график в HTML и PNG (если доступно).

    Args:
        fig: Plotly figure
        output_dir: Директория для сохранения
        name: Имя файла (без расширения)
    """
    global CAN_EXPORT_PNG

    # Создаём директорию если не существует
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Всегда сохраняем HTML
    html_path = os.path.join(output_dir, f'{name}.html')
    fig.write_html(html_path)

    # Пробуем сохранить PNG
    if CAN_EXPORT_PNG:
        try:
            png_path = os.path.join(output_dir, f'{name}.png')
            fig.write_image(png_path, scale=2)
            print(f"[INFO] Сохранён график: {name}.html, {name}.png")
        except Exception as e:
            CAN_EXPORT_PNG = False
            print(f"[WARNING] Экспорт PNG недоступен (требуется Chrome): {e}")
            print(f"[INFO] Сохранён график: {name}.html")
    else:
        print(f"[INFO] Сохранён график: {name}.html")


def extract_model_label_from_filename(filename: str) -> str:
    """
    Извлекает метку модели из имени файла benchmark_summary.

    Формат: benchmark_summary_{model}_{metadata}_{timestamp}.json
    или: benchmark_summary_{model}_{timestamp}.json

    Args:
        filename: Имя файла (например, "benchmark_summary_Qwen_Qwen3-0.6B_fp16_20241219_123456.json")

    Returns:
        Метка модели (например, "Qwen_Qwen3-0.6B_fp16" или "Qwen_Qwen3-0.6B")
    """
    # Убираем расширение и префикс
    name = Path(filename).stem
    prefix = "benchmark_summary_"
    if name.startswith(prefix):
        name = name[len(prefix):]

    # Убираем timestamp в конце (формат: _YYYYMMDD_HHMMSS)
    # Паттерн: _цифры_цифры в конце
    timestamp_pattern = r'_\d{8}_\d{6}$'
    name = re.sub(timestamp_pattern, '', name)

    return name


def load_benchmark_summary(file_path: str, model_label: str = None) -> pd.DataFrame:
    """
    Загружает данные из JSON файла и преобразует в DataFrame.

    Args:
        file_path: Путь к файлу benchmark_summary_*.json
        model_label: Опциональная метка модели для добавления в DataFrame

    Returns:
        DataFrame с данными бенчмарка
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Если метка не указана, извлекаем из имени файла
    if model_label is None:
        model_label = extract_model_label_from_filename(file_path)

    # Преобразуем вложенные данные в плоскую структуру
    records = []
    for entry in data:
        record = {
            'model': entry['model'],
            'model_label': model_label,
            'dataset': entry['dataset'],
            'max_concurrency': entry['max_concurrency'],
            'input_len': entry['input_len'],
            'output_len': entry['output_len'],
        }
        # Добавляем метрики из result
        result = entry['result']
        record['output_throughput'] = result['output_throughput']
        record['total_token_throughput'] = result['total_token_throughput']
        record['request_throughput'] = result['request_throughput']
        record['mean_ttft_ms'] = result['mean_ttft_ms']
        record['median_ttft_ms'] = result['median_ttft_ms']
        record['p99_ttft_ms'] = result['p99_ttft_ms']
        record['mean_tpot_ms'] = result['mean_tpot_ms']
        record['median_tpot_ms'] = result['median_tpot_ms']
        record['p99_tpot_ms'] = result['p99_tpot_ms']
        record['mean_itl_ms'] = result['mean_itl_ms']
        record['median_itl_ms'] = result['median_itl_ms']
        record['p99_itl_ms'] = result['p99_itl_ms']
        record['mean_e2el_ms'] = result['mean_e2el_ms']
        record['median_e2el_ms'] = result['median_e2el_ms']
        record['p99_e2el_ms'] = result['p99_e2el_ms']
        record['duration'] = result['duration']
        record['completed'] = result['completed']
        record['total_input_tokens'] = result['total_input_tokens']
        record['total_output_tokens'] = result['total_output_tokens']

        records.append(record)

    return pd.DataFrame(records)


def plot_performance_comparison(
    df: pd.DataFrame,
    output_dir: str,
    color_by: str = 'dataset',
    facet_col: str = None,
    title_suffix: str = ''
) -> None:
    """
    Строит график сравнения производительности LLM.

    X: max_concurrency
    Y: output_throughput (токенов/сек)

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения графиков
        color_by: Колонка для группировки по цвету ('dataset' или 'model_label')
        facet_col: Колонка для разбиения на панели (например, 'dataset')
        title_suffix: Суффикс для заголовка графика
    """
    title = format_chart_title(
        'Сравнение производительности LLM', title_suffix
    )

    color_labels = {
        'dataset': 'Датасет',
        'model_label': 'Модель',
        'model_dataset': 'Модель / Датасет'
    }
    color_label = color_labels.get(color_by, color_by)

    fig = px.line(
        df,
        x='max_concurrency',
        y='output_throughput',
        color=color_by,
        facet_col=facet_col,
        markers=True,
        title=title,
        labels={
            'max_concurrency': 'Параллельные запросы (Concurrency)',
            'output_throughput': 'Генерация токенов (токен/сек)',
            color_by: color_label
        },
        color_discrete_sequence=CHART_COLORS
    )

    height = 600 if facet_col else 500

    fig.update_layout(
        font=dict(size=14, family=CHART_STYLE['font_family']),
        title=dict(
            font=dict(size=16), x=0.5, xanchor='center'
        ),
        height=height,
        width=1200,
        plot_bgcolor=CHART_STYLE['plot_bgcolor'],
        paper_bgcolor=CHART_STYLE['paper_bgcolor'],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(l=80, r=40, t=100, b=100),
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=sorted(df['max_concurrency'].unique()),
        tickangle=0,
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_yaxes(
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=10))

    save_figure(fig, output_dir, 'performance_comparison')


def plot_ttft_comparison(
    df: pd.DataFrame,
    output_dir: str,
    color_by: str = 'dataset',
    facet_col: str = None,
    title_suffix: str = ''
) -> None:
    """
    Строит график Time to First Token (TTFT) vs Concurrency.

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения графиков
        color_by: Колонка для группировки по цвету
        facet_col: Колонка для разбиения на панели
        title_suffix: Суффикс для заголовка графика
    """
    title = format_chart_title(
        'Время до первого токена (TTFT) vs Параллелизм',
        title_suffix,
    )

    color_labels = {
        'dataset': 'Датасет',
        'model_label': 'Модель',
        'model_dataset': 'Модель / Датасет'
    }
    color_label = color_labels.get(color_by, color_by)

    fig = px.line(
        df,
        x='max_concurrency',
        y='mean_ttft_ms',
        color=color_by,
        facet_col=facet_col,
        markers=True,
        title=title,
        labels={
            'max_concurrency': 'Параллельные запросы (Concurrency)',
            'mean_ttft_ms': 'Среднее TTFT (мс)',
            color_by: color_label
        },
        color_discrete_sequence=CHART_COLORS
    )

    height = 600 if facet_col else 500

    fig.update_layout(
        font=dict(size=14, family=CHART_STYLE['font_family']),
        title=dict(
            font=dict(size=16), x=0.5, xanchor='center'
        ),
        height=height,
        width=1200,
        plot_bgcolor=CHART_STYLE['plot_bgcolor'],
        paper_bgcolor=CHART_STYLE['paper_bgcolor'],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(l=80, r=40, t=100, b=100),
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=sorted(df['max_concurrency'].unique()),
        tickangle=0,
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_yaxes(
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=10))

    save_figure(fig, output_dir, 'ttft_comparison')


def plot_tpot_comparison(
    df: pd.DataFrame,
    output_dir: str,
    color_by: str = 'dataset',
    facet_col: str = None,
    title_suffix: str = ''
) -> None:
    """
    Строит график Time per Output Token (TPOT) vs Concurrency.

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения графиков
        color_by: Колонка для группировки по цвету
        facet_col: Колонка для разбиения на панели
        title_suffix: Суффикс для заголовка графика
    """
    title = format_chart_title(
        'Время на выходной токен (TPOT) vs Параллелизм',
        title_suffix,
    )

    color_labels = {
        'dataset': 'Датасет',
        'model_label': 'Модель',
        'model_dataset': 'Модель / Датасет'
    }
    color_label = color_labels.get(color_by, color_by)

    fig = px.line(
        df,
        x='max_concurrency',
        y='mean_tpot_ms',
        color=color_by,
        facet_col=facet_col,
        markers=True,
        title=title,
        labels={
            'max_concurrency': 'Параллельные запросы (Concurrency)',
            'mean_tpot_ms': 'Среднее TPOT (мс)',
            color_by: color_label
        },
        color_discrete_sequence=CHART_COLORS
    )

    height = 600 if facet_col else 500

    fig.update_layout(
        font=dict(size=14, family=CHART_STYLE['font_family']),
        title=dict(
            font=dict(size=16), x=0.5, xanchor='center'
        ),
        height=height,
        width=1200,
        plot_bgcolor=CHART_STYLE['plot_bgcolor'],
        paper_bgcolor=CHART_STYLE['paper_bgcolor'],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(l=80, r=40, t=100, b=100),
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=sorted(df['max_concurrency'].unique()),
        tickangle=0,
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_yaxes(
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=10))

    save_figure(fig, output_dir, 'tpot_comparison')


def plot_latency_percentiles(
    df: pd.DataFrame,
    output_dir: str,
    facet_by: str = 'dataset',
    title_suffix: str = ''
) -> None:
    """
    Строит график сравнения задержек (mean, median, p99) для TTFT.

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения графиков
        facet_by: Колонка для разделения на панели
        title_suffix: Суффикс для заголовка графика
    """
    title = format_chart_title(
        'Распределение задержек TTFT по перцентилям',
        title_suffix,
    )

    facet_labels = {
        'dataset': 'Датасет',
        'model_label': 'Модель',
        'model_dataset': 'Модель / Датасет'
    }
    facet_label = facet_labels.get(facet_by, facet_by)

    # Преобразуем данные для grouped bar chart
    melted = df.melt(
        id_vars=['max_concurrency', facet_by],
        value_vars=['mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms'],
        var_name='metric',
        value_name='value'
    )

    # Переименовываем метрики для читаемости
    metric_names = {
        'mean_ttft_ms': 'Среднее',
        'median_ttft_ms': 'Медиана',
        'p99_ttft_ms': 'P99'
    }
    melted['metric'] = melted['metric'].map(metric_names)

    # Цвета для метрик перцентилей
    percentile_colors = ['#2E86AB', '#F39237', '#E94F37']  # Синий, Оранжевый, Красный

    fig = px.bar(
        melted,
        x='max_concurrency',
        y='value',
        color='metric',
        barmode='group',
        facet_col=facet_by,
        title=title,
        labels={
            'max_concurrency': 'Параллельные запросы',
            'value': 'Задержка (мс)',
            'metric': 'Метрика',
            facet_by: facet_label
        },
        color_discrete_sequence=percentile_colors
    )

    fig.update_layout(
        font=dict(size=14, family=CHART_STYLE['font_family']),
        title=dict(
            font=dict(size=16), x=0.5, xanchor='center'
        ),
        height=550,
        width=1200,
        plot_bgcolor=CHART_STYLE['plot_bgcolor'],
        paper_bgcolor=CHART_STYLE['paper_bgcolor'],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(l=80, r=40, t=100, b=100),
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)

    fig.update_xaxes(
        tickangle=0,
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_yaxes(
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    save_figure(fig, output_dir, 'latency_percentiles')


def plot_throughput_bar(
    df: pd.DataFrame,
    output_dir: str,
    color_by: str = 'dataset',
    facet_col: str = None,
    title_suffix: str = ''
) -> None:
    """
    Строит столбчатую диаграмму throughput по concurrency.

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения графиков
        color_by: Колонка для группировки по цвету
        facet_col: Колонка для разбиения на панели
        title_suffix: Суффикс для заголовка графика
    """
    title = format_chart_title(
        'Пропускная способность генерации по уровню параллелизма',
        title_suffix,
    )

    color_labels = {
        'dataset': 'Датасет',
        'model_label': 'Модель',
        'model_dataset': 'Модель / Датасет'
    }
    color_label = color_labels.get(color_by, color_by)

    fig = px.bar(
        df,
        x='max_concurrency',
        y='output_throughput',
        color=color_by,
        facet_col=facet_col,
        barmode='group',
        title=title,
        labels={
            'max_concurrency': 'Параллельные запросы (Concurrency)',
            'output_throughput': 'Генерация токенов (токен/сек)',
            color_by: color_label
        },
        text_auto='.0f',
        color_discrete_sequence=CHART_COLORS
    )

    height = 600 if facet_col else 500

    fig.update_layout(
        font=dict(size=14, family=CHART_STYLE['font_family']),
        title=dict(
            font=dict(size=16), x=0.5, xanchor='center'
        ),
        height=height,
        width=1200,
        plot_bgcolor=CHART_STYLE['plot_bgcolor'],
        paper_bgcolor=CHART_STYLE['paper_bgcolor'],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(l=80, r=40, t=100, b=120),
    )

    fig.update_xaxes(
        tickangle=0,
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_yaxes(
        gridcolor=CHART_STYLE['grid_color'],
        linecolor=CHART_STYLE['line_color'],
        showline=True
    )

    fig.update_traces(textposition='outside')

    save_figure(fig, output_dir, 'throughput_bar')


def plot_combined_metrics(
    df: pd.DataFrame,
    output_dir: str,
    group_by: str = 'dataset',
    facet_by: str = None,
    title_suffix: str = ''
) -> None:
    """
    Строит комбинированный график с несколькими метриками.

    Улучшенная версия с чётким разделением по датасетам и понятными цветами.

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения графиков
        group_by: Колонка для группировки (цвет линий)
        facet_by: Колонка для разбиения на строки (например, 'dataset')
        title_suffix: Суффикс для заголовка графика
    """
    title = 'Комплексный анализ производительности LLM'
    if title_suffix:
        title = f'{title} ({title_suffix})'

    groups = df[group_by].unique()
    concurrency_values = sorted(df['max_concurrency'].unique())

    # Определяем метрики и их описания
    metrics = [
        ('output_throughput', 'Throughput\n(токен/сек)', '↑ лучше'),
        ('mean_ttft_ms', 'TTFT\n(мс)', '↓ лучше'),
        ('mean_tpot_ms', 'TPOT\n(мс)', '↓ лучше'),
        ('mean_e2el_ms', 'E2E Latency\n(мс)', '↓ лучше'),
    ]

    if facet_by:
        facet_values = list(df[facet_by].unique())
        n_rows = len(facet_values)
        n_cols = len(metrics)

        # Создаём заголовки только для первой строки (названия метрик)
        subplot_titles = []
        for row_idx in range(n_rows):
            for metric_name, metric_label, direction in metrics:
                if row_idx == 0:
                    subplot_titles.append(f"{metric_label} {direction}".replace('\n', ' '))
                else:
                    subplot_titles.append('')  # Пустые заголовки для остальных строк

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )

        for row_idx, facet_value in enumerate(facet_values, start=1):
            facet_df = df[df[facet_by] == facet_value]

            for col_idx, (metric_name, metric_label, direction) in enumerate(metrics, start=1):
                for i, group in enumerate(groups):
                    subset = facet_df[facet_df[group_by] == group]
                    if len(subset) == 0:
                        continue

                    color = CHART_COLORS[i % len(CHART_COLORS)]
                    show_legend = (row_idx == 1 and col_idx == 1)

                    fig.add_trace(
                        go.Scatter(
                            x=subset['max_concurrency'],
                            y=subset[metric_name],
                            mode='lines+markers',
                            name=group,
                            line=dict(color=color, width=3),
                            marker=dict(size=8, symbol='circle'),
                            legendgroup=group,
                            showlegend=show_legend,
                            hovertemplate=(
                                f'<b>{group}</b><br>'
                                f'Concurrency: %{{x}}<br>'
                                f'{metric_label.replace(chr(10), " ")}: %{{y:.1f}}<br>'
                                '<extra></extra>'
                            )
                        ),
                        row=row_idx, col=col_idx
                    )

        # Высота: 350px на строку + отступы
        height = 350 * n_rows + 180

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22, family='Arial Black'),
                x=0.5,
                xanchor='center',
                y=0.98
            ),
            font=dict(size=13, family=CHART_STYLE['font_family']),
            height=height,
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.045,
                xanchor="center",
                x=0.5,
                font=dict(size=14),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=CHART_STYLE['grid_color'],
                borderwidth=1
            ),
            plot_bgcolor=CHART_STYLE['plot_bgcolor'],
            paper_bgcolor=CHART_STYLE['paper_bgcolor'],
            margin=dict(l=60, r=40, t=140, b=60)
        )

        # Стилизация осей
        fig.update_xaxes(
            tickmode='array',
            tickvals=concurrency_values,
            title_text='Concurrency',
            title_font=dict(size=11),
            tickfont=dict(size=10),
            tickangle=0,
            gridcolor=CHART_STYLE['grid_color'],
            linecolor=CHART_STYLE['line_color'],
            showline=True
        )

        fig.update_yaxes(
            gridcolor=CHART_STYLE['grid_color'],
            linecolor=CHART_STYLE['line_color'],
            showline=True,
            tickfont=dict(size=10)
        )

        # Стилизация заголовков subplot'ов (названия метрик)
        for annotation in fig['layout']['annotations']:
            if annotation['text']:
                annotation['font'] = dict(size=14, family=CHART_STYLE['font_family'], color='#333333')

        # Добавляем подписи датасетов по центру сверху каждой строки
        positions = [1.02, 0.65, 0.28]
        for row_idx, facet_value in enumerate(facet_values):
            row_top_y = positions[row_idx] if row_idx < len(positions) else 1.0 - row_idx * 0.35

            fig.add_annotation(
                text=f"<b>{facet_value}</b>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=row_top_y,
                showarrow=False,
                font=dict(size=15, family=CHART_STYLE['font_family'], color='#1a1a1a'),
                xanchor='center',
                yanchor='bottom'
            )

    else:
        # Простой 2x2 layout для одной модели/датасета
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m[1].replace('\n', ' ') for m in metrics],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        for i, group in enumerate(groups):
            subset = df[df[group_by] == group]
            color = CHART_COLORS[i % len(CHART_COLORS)]

            for idx, (metric_name, metric_label, direction) in enumerate(metrics):
                row = idx // 2 + 1
                col = idx % 2 + 1

                fig.add_trace(
                    go.Scatter(
                        x=subset['max_concurrency'],
                        y=subset[metric_name],
                        mode='lines+markers',
                        name=group,
                        line=dict(color=color, width=3),
                        marker=dict(size=10, symbol='circle'),
                        legendgroup=group,
                        showlegend=(idx == 0),
                        hovertemplate=(
                            f'<b>{group}</b><br>'
                            f'Concurrency: %{{x}}<br>'
                            f'{metric_label.replace(chr(10), " ")}: %{{y:.1f}}<br>'
                            '<extra></extra>'
                        )
                    ),
                    row=row, col=col
                )

        height = 1000

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22, family='Arial Black'),
                x=0.5,
                y=0.995,
                xanchor='center'
            ),
            font=dict(size=13, family=CHART_STYLE['font_family']),
            height=height,
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="center",
                x=0.5,
                font=dict(size=14),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=CHART_STYLE['grid_color'],
                borderwidth=1
            ),
            plot_bgcolor=CHART_STYLE['plot_bgcolor'],
            paper_bgcolor=CHART_STYLE['paper_bgcolor']
        )

        fig.update_xaxes(
            tickmode='array',
            tickvals=concurrency_values,
            title_text='Concurrency',
            tickangle=0,
            gridcolor=CHART_STYLE['grid_color'],
            linecolor=CHART_STYLE['line_color'],
            showline=True
        )

        fig.update_yaxes(
            gridcolor=CHART_STYLE['grid_color'],
            linecolor=CHART_STYLE['line_color'],
            showline=True
        )

        # Стилизация заголовков subplot'ов
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, family=CHART_STYLE['font_family'], color='#333333')

    save_figure(fig, output_dir, 'combined_metrics')


def plot_summary_table(
    df: pd.DataFrame,
    output_dir: str,
    group_by: str = 'dataset',
    title_suffix: str = '',
    show_relative: bool = False
) -> None:
    """
    Создаёт сводную таблицу с ключевыми метриками.

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения
        group_by: Колонка для группировки
        title_suffix: Суффикс для заголовка
        show_relative: Показывать относительные значения (лучшая модель = 100%)
    """
    title = format_chart_title(
        'Сводная таблица результатов бенчмарка',
        title_suffix,
    )

    group_labels = {
        'dataset': 'Датасет',
        'model_label': 'Модель',
        'model_dataset': 'Модель / Датасет'
    }
    group_label = group_labels.get(group_by, group_by)

    # Группируем и выбираем лучшие показатели
    summary = df.groupby(group_by).agg({
        'output_throughput': ['max', 'min'],
        'mean_ttft_ms': ['min', 'max'],
        'mean_tpot_ms': ['min', 'max'],
        'max_concurrency': lambda x: x[df.loc[x.index, 'output_throughput'].idxmax()]
    }).round(2)

    summary.columns = [
        'Макс. throughput (tok/s)', 'Мин. throughput (tok/s)',
        'Мин. TTFT (мс)', 'Макс. TTFT (мс)',
        'Мин. TPOT (мс)', 'Макс. TPOT (мс)',
        'Оптим. concurrency'
    ]

    if show_relative and len(summary) > 1:
        # Вычисляем относительные значения
        # Для throughput: больше лучше, лучшая модель = 100%
        # Для latency: меньше лучше, лучшая модель = 100%

        # Макс. throughput: больше лучше
        best_max_throughput = summary['Макс. throughput (tok/s)'].max()
        summary['Макс. throughput (%)'] = (
            summary['Макс. throughput (tok/s)'] / best_max_throughput * 100
        ).round(1)

        # Мин. throughput: больше лучше
        best_min_throughput = summary['Мин. throughput (tok/s)'].max()
        summary['Мин. throughput (%)'] = (
            summary['Мин. throughput (tok/s)'] / best_min_throughput * 100
        ).round(1)

        # Мин. TTFT: меньше лучше
        best_min_ttft = summary['Мин. TTFT (мс)'].min()
        summary['Мин. TTFT (%)'] = (
            best_min_ttft / summary['Мин. TTFT (мс)'] * 100
        ).round(1)

        # Макс. TTFT: меньше лучше
        best_max_ttft = summary['Макс. TTFT (мс)'].min()
        summary['Макс. TTFT (%)'] = (
            best_max_ttft / summary['Макс. TTFT (мс)'] * 100
        ).round(1)

        # Мин. TPOT: меньше лучше
        best_min_tpot = summary['Мин. TPOT (мс)'].min()
        summary['Мин. TPOT (%)'] = (
            best_min_tpot / summary['Мин. TPOT (мс)'] * 100
        ).round(1)

        # Макс. TPOT: меньше лучше
        best_max_tpot = summary['Макс. TPOT (мс)'].min()
        summary['Макс. TPOT (%)'] = (
            best_max_tpot / summary['Макс. TPOT (мс)'] * 100
        ).round(1)

        # Форматируем значения с процентами
        def format_with_percent(value, percent):
            return f"{value} ({percent}%)"

        formatted_values = {
            'Макс. throughput': [
                format_with_percent(v, p) for v, p in
                zip(summary['Макс. throughput (tok/s)'], summary['Макс. throughput (%)'])
            ],
            'Мин. throughput': [
                format_with_percent(v, p) for v, p in
                zip(summary['Мин. throughput (tok/s)'], summary['Мин. throughput (%)'])
            ],
            'Мин. TTFT': [
                format_with_percent(v, p) for v, p in
                zip(summary['Мин. TTFT (мс)'], summary['Мин. TTFT (%)'])
            ],
            'Макс. TTFT': [
                format_with_percent(v, p) for v, p in
                zip(summary['Макс. TTFT (мс)'], summary['Макс. TTFT (%)'])
            ],
            'Мин. TPOT': [
                format_with_percent(v, p) for v, p in
                zip(summary['Мин. TPOT (мс)'], summary['Мин. TPOT (%)'])
            ],
            'Макс. TPOT': [
                format_with_percent(v, p) for v, p in
                zip(summary['Макс. TPOT (мс)'], summary['Макс. TPOT (%)'])
            ],
        }

        # Создаём таблицу с форматированными значениями
        header_values = [
            group_label,
            'Макс. throughput<br>(tok/s)',
            'Мин. throughput<br>(tok/s)',
            'Мин. TTFT<br>(мс)',
            'Макс. TTFT<br>(мс)',
            'Мин. TPOT<br>(мс)',
            'Макс. TPOT<br>(мс)',
            'Оптим.<br>concurrency'
        ]

        cell_values = [
            summary.index,
            formatted_values['Макс. throughput'],
            formatted_values['Мин. throughput'],
            formatted_values['Мин. TTFT'],
            formatted_values['Макс. TTFT'],
            formatted_values['Мин. TPOT'],
            formatted_values['Макс. TPOT'],
            summary['Оптим. concurrency']
        ]
    else:
        header_values = [group_label] + list(summary.columns)
        cell_values = [summary.index] + [summary[col] for col in summary.columns]

    # Создаём таблицу plotly
    fig = go.Figure(data=[go.Table(
        columnwidth=[2.5, 1, 1, 1, 1, 1, 1, 1],
        header=dict(
            values=header_values,
            fill_color=CHART_COLORS[0],
            font=dict(
                color='white', size=14,
                family=CHART_STYLE['font_family'],
            ),
            align=['left']
            + ['center'] * (len(header_values) - 1),
            height=40,
            line_color=CHART_STYLE['line_color']
        ),
        cells=dict(
            values=cell_values,
            fill_color=[
                [CHART_STYLE['plot_bgcolor'], 'white']
                * (len(summary) // 2 + 1)
            ][:len(summary)],
            font=dict(
                size=13,
                family=CHART_STYLE['font_family'],
            ),
            align=['left']
            + ['center'] * (len(header_values) - 1),
            height=35,
            line_color=CHART_STYLE['grid_color']
        )
    )])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=16,
                family=CHART_STYLE['font_family'],
            ),
            x=0.5,
            xanchor='center'
        ),
        height=350 if show_relative else 300,
        width=1200,
        paper_bgcolor=CHART_STYLE['paper_bgcolor'],
        margin=dict(l=20, r=20, t=100, b=20)
    )

    save_figure(fig, output_dir, 'summary_table')


def generate_model_plots(df: pd.DataFrame, output_dir: str, model_label: str) -> None:
    """
    Генерирует все графики для одной модели.

    Args:
        df: DataFrame с данными одной модели
        output_dir: Базовая директория для графиков
        model_label: Метка модели для именования поддиректории
    """
    model_dir = os.path.join(output_dir, model_label)

    print(f"\n[INFO] Генерация графиков для модели: {model_label}")

    plot_performance_comparison(df, model_dir, color_by='dataset', title_suffix=model_label)
    plot_ttft_comparison(df, model_dir, color_by='dataset', title_suffix=model_label)
    plot_tpot_comparison(df, model_dir, color_by='dataset', title_suffix=model_label)
    plot_latency_percentiles(df, model_dir, facet_by='dataset', title_suffix=model_label)
    plot_throughput_bar(df, model_dir, color_by='dataset', title_suffix=model_label)
    plot_combined_metrics(df, model_dir, group_by='dataset', title_suffix=model_label)
    plot_summary_table(df, model_dir, group_by='dataset', title_suffix=model_label)


def generate_comparison_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Генерирует сравнительные графики для всех моделей.

    Графики разбиваются на панели по датасетам, модели отображаются разными цветами.

    Args:
        df: DataFrame с данными всех моделей
        output_dir: Базовая директория для графиков
    """
    comparison_dir = os.path.join(output_dir, 'comparison')

    print(f"\n[INFO] Генерация сравнительных графиков для всех моделей")

    # Для сравнительных графиков: панели по датасетам, цвет по моделям
    plot_performance_comparison(
        df, comparison_dir, color_by='model_label', facet_col='dataset'
    )
    plot_ttft_comparison(
        df, comparison_dir, color_by='model_label', facet_col='dataset'
    )
    plot_tpot_comparison(
        df, comparison_dir, color_by='model_label', facet_col='dataset'
    )
    plot_latency_percentiles(
        df, comparison_dir, facet_by='model_label'
    )
    plot_throughput_bar(
        df, comparison_dir, color_by='model_label', facet_col='dataset'
    )
    plot_combined_metrics(
        df, comparison_dir, group_by='model_label', facet_by='dataset'
    )
    plot_summary_table(
        df, comparison_dir, group_by='model_label',
        show_relative=False
    )


def find_latest_summary_per_model(results_dir: Path) -> dict[str, Path]:
    """
    Находит последний summary файл для каждой модели.

    Args:
        results_dir: Директория с результатами

    Returns:
        Словарь {model_label: path_to_latest_summary}
    """
    summary_files = list(results_dir.glob('benchmark_summary_*.json'))

    if not summary_files:
        return {}

    # Группируем файлы по метке модели
    model_files = {}
    for f in summary_files:
        model_label = extract_model_label_from_filename(str(f))
        if model_label not in model_files:
            model_files[model_label] = []
        model_files[model_label].append(f)

    # Выбираем последний файл для каждой модели
    latest_files = {}
    for model_label, files in model_files.items():
        latest_files[model_label] = max(files, key=lambda x: x.stat().st_mtime)

    return latest_files


def main():
    """
    Главная функция визуализации.
    """
    print("="*60)
    print("  vLLM Benchmark Visualization")
    print("="*60)

    # Определяем пути
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    plots_dir = script_dir / 'plots'

    # Создаём директорию для графиков
    plots_dir.mkdir(exist_ok=True)

    # Если указан конкретный файл, работаем только с ним
    if len(sys.argv) > 1:
        summary_file = Path(sys.argv[1])
        print(f"[INFO] Загрузка данных из: {summary_file}")
        df = load_benchmark_summary(str(summary_file))
        model_label = extract_model_label_from_filename(str(summary_file))

        print(f"[INFO] Загружено записей: {len(df)}")
        print(f"[INFO] Модель: {model_label}")
        print(f"[INFO] Датасеты: {df['dataset'].unique().tolist()}")
        print(f"[INFO] Concurrency: {sorted(df['max_concurrency'].unique().tolist())}")

        # Генерируем графики только для этой модели
        generate_model_plots(df, str(plots_dir), model_label)
    else:
        # Находим все summary файлы (последние для каждой модели)
        latest_files = find_latest_summary_per_model(results_dir)

        if not latest_files:
            print("[ERROR] Файлы benchmark_summary_*.json не найдены в ./results/")
            sys.exit(1)

        print(f"[INFO] Найдено моделей: {len(latest_files)}")
        for model_label, file_path in latest_files.items():
            print(f"       - {model_label}: {file_path.name}")

        # Загружаем данные всех моделей
        all_dfs = []
        for model_label, file_path in latest_files.items():
            df = load_benchmark_summary(str(file_path), model_label)
            all_dfs.append(df)

            print(f"\n[INFO] Модель {model_label}:")
            print(f"       Записей: {len(df)}")
            print(f"       Датасеты: {df['dataset'].unique().tolist()}")

        # Объединяем все данные
        combined_df = pd.concat(all_dfs, ignore_index=True)

        print(f"\n[INFO] Всего записей: {len(combined_df)}")
        print(f"[INFO] Concurrency: {sorted(combined_df['max_concurrency'].unique().tolist())}")

        # Генерируем графики для каждой модели отдельно
        for model_label in latest_files.keys():
            model_df = combined_df[combined_df['model_label'] == model_label]
            generate_model_plots(model_df, str(plots_dir), model_label)

        # Генерируем сравнительные графики (если больше одной модели)
        if len(latest_files) > 1:
            generate_comparison_plots(combined_df, str(plots_dir))

    print("\n" + "="*60)
    print(f"  Графики сохранены в: {plots_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
