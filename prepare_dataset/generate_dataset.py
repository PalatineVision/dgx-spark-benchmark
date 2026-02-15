import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# т.к. будем тестировать модели серии Qwen возьмём токенизатор этой модели, для подсчёта длины контекста
# может понадобиться VP* для загрузки конфига токенизатора
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# neural-bridge/rag-dataset-12000
splits = {'train': 'data/train-00000-of-00001-9df3a936e1f63191.parquet',
          'test': 'data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'}
rag_df = pd.read_parquet("hf://datasets/neural-bridge/rag-dataset-12000/" + splits["train"])

# расширим контекст для тестов с длинным контекстом
rag_df["long_context"] = rag_df["context"].apply(lambda x: x * 10)

# Предварительно отобрано несколько записей для бенчмарков
# для RAG бенчмарка отобраны данные с контекстом от 6000 до 9000 символов
# для chat бенчмарка отобраны данные с короткими промптами менее 50 символов
long_rag_selected = [247, 828, 862, 1152, 1326, 1851, 2222, 2526, 2934, 2953, 3193, 3241, 3607, 3775, 4001, 4138, 4637,
                     6010, 6256, 6523, 7302, 7410, 7449, 7612, 7952, 8435, 8619, 8890]

middle_rag_selected = [8364, 7171, 4257, 7454, 6888, 3550, 9557, 7586, 1628, 7280, 4010, 7076, 114, 7213, 4105, 6185,
                       2939, 8002, 8341, 6863, 301, 5523, 3191, 6413, 7006, 2076, 2447, 1021, 680, 6945, 936, 8111,
                       4798, 671, 7915, 6965, 231, 1330, 8528, 6150, 4978, 2750, 2122, 4761, 5420, 7075, 5499, 534,
                       8784, 3173]

chat_selected = [3452, 828, 7498, 4425, 2712, 4337, 5683, 5345, 1837, 1793, 4636, 2784, 5683, 2154, 8110, 2276, 6718,
                 6056, 4618, 8732, 1929, 4875, 2197, 4363, 6561, 8958, 4337, 8655, 4750, 4952, 9533, 69, 9193, 1765,
                 3904, 7615, 2276, 800, 4932, 1706, 2459, 4346, 1922, 2712, 4242, 5018, 28, 5865, 4223, 9193]

print("="*15, f"RAG long context ({len(long_rag_selected)} items)", "="*15)
print("="*15, f"RAG middle context ({len(middle_rag_selected)} items)", "="*15)
print("="*15, f"Chat data ({len(chat_selected)} items)", "="*15)

# сохраняем данные в формате, пригодном для vLLM бенчмарка
dataset_path = Path("dataset")
dataset_path.mkdir(parents=True, exist_ok=True)

with open(dataset_path / "long_context_rag.jsonl", "w", encoding="utf-8") as rag_dataset_file:
    for idx, raw in rag_df.loc[long_rag_selected].iterrows():
        rag_dataset_file.write(
            json.dumps({"prompt": f"Context:\n{raw['context']}\n\nUser question:\n{raw['question']}\n"}) + "\n"
        )

with open(dataset_path / "middle_context_rag.jsonl", "w", encoding="utf-8") as rag_dataset_file:
    for idx, raw in rag_df.loc[middle_rag_selected].iterrows():
        rag_dataset_file.write(
            json.dumps({"prompt": f"Context:\n{raw['context']}\n\nUser question:\n{raw['question']}\n"}) + "\n"
        )

with open(dataset_path.absolute() / "low_context_chat.jsonl", "w", encoding="utf-8") as chat_dataset_file:
    for idx, raw in rag_df.loc[chat_selected].iterrows():
        chat_dataset_file.write(
            json.dumps({"prompt": f"{raw['question']}"}) + "\n"
        )
