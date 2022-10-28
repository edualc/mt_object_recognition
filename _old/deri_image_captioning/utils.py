import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from transformers import RobertaTokenizer
from transformers.file_utils import PaddingStrategy
from torch.utils.data.dataset import Dataset


class SingleContextDataset(Dataset):
    def __init__(
            self,
            tokenizer: RobertaTokenizer,
            caption: List[str],
            block_size: int,
            batch_size:int
    ):
        batch_encoding = tokenizer([caption], add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long), 'labels': torch.tensor(e, dtype=torch.long)} for e in self.examples]*batch_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class DummyCollator:

    tokenizer: RobertaTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        return batch