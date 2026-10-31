import os
import json
import torch
from torch.utils.data import Dataset
from pyvi import ViTokenizer
from transformers import AutoTokenizer


LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class UITVSFCDataset(Dataset):
    """
    UIT-VSFC Dataset with Vietnamese word segmentation and PhoBERT tokenization.
    Supports caching pre-segmented text to avoid redundant segmentation overhead.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=96,
        cache_dir="data/cache",
        split_name=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        if split_name is None:
            split_name = os.path.splitext(os.path.basename(data_path))[0]

        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, f"{split_name}_segmented.json")

        if os.path.exists(cached_file):
            with open(cached_file, "r", encoding="utf-8") as f:
                raw_items = json.load(f)
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_items = json.load(f)

            # Segment Vietnamese words
            for item in raw_items:
                raw_sentence = item["Sentence"]
                seg_sentence = ViTokenizer.tokenize(raw_sentence)
                item["Segmented_Sentence"] = seg_sentence

            with open(cached_file, "w", encoding="utf-8") as f:
                json.dump(raw_items, f, ensure_ascii=False, indent=2)

        for item in raw_items:
            sentiment_str = item["Sentiment"].lower()
            label = LABEL_MAP[sentiment_str]
            self.data.append({
                "sentence": item["Segmented_Sentence"],
                "raw_sentence": item["Sentence"],
                "label": label,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    def get_labels(self):
        return [item["label"] for item in self.data]


class Collator:
    """
    Collate function to batch and pad input tensors for PhoBERT.
    """

    def __init__(self, tokenizer, max_length=96):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item["sentence"] for item in batch]
        labels = [item["label"] for item in batch]

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encodings["labels"] = torch.tensor(labels, dtype=torch.long)
        return encodings
