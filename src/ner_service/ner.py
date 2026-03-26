from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .utils import drop_invalid_spans, remove_overlaps_prefer_longest

try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForTokenClassification = None
    AutoTokenizer = None


SpanTuple = Tuple[int, int, str]


REGEX_PATTERNS = {
    "Email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "Номер телефона": r"(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
    "СНИЛС клиента": r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}\b",
    "Паспортные данные": r"\b\d{2}\s?\d{2}\s?\d{6}\b",
    "Сведения об ИНН": r"\b\d{10}(?:\d{2})?\b",
    "Номер карты": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "Номер банковского счета": r"\b\d{20}\b",
}

STRIP_TRAILING = set(".,;:!)")
STRIP_LEADING = set(" \t\n")


@dataclass
class LoadedHFModel:
    tokenizer: any
    model: any
    device: str


class NERService:
    def __init__(self, model_path: Path | str, device: str = "cpu", max_length: int = 256):
        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length
        self._hf: Optional[LoadedHFModel] = None

        if self.model_path.exists() and any(self.model_path.iterdir()):
            self._hf = self._load_hf_model()

    @property
    def mode(self) -> str:
        return "huggingface" if self._hf is not None else "regex"

    def _load_hf_model(self) -> Optional[LoadedHFModel]:
        if AutoTokenizer is None or AutoModelForTokenClassification is None:
            return None
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        actual_device = self.device
        if torch is not None:
            if actual_device.startswith("cuda") and not torch.cuda.is_available():
                actual_device = "cpu"
            model.to(actual_device)
        model.eval()
        return LoadedHFModel(tokenizer=tokenizer, model=model, device=actual_device)

    def predict(self, text: str) -> List[SpanTuple]:
        if not text:
            return []
        if self._hf is None:
            return self._predict_regex(text)
        return self._predict_hf(text)

    def _predict_regex(self, text: str) -> List[SpanTuple]:
        spans: List[SpanTuple] = []
        for label, pattern in REGEX_PATTERNS.items():
            for match in re.finditer(pattern, text):
                spans.append((match.start(), match.end(), label))
        spans = self._postprocess(text, spans)
        return remove_overlaps_prefer_longest(drop_invalid_spans(text, spans))

    def _predict_hf(self, text: str) -> List[SpanTuple]:
        hf = self._hf
        assert hf is not None

        encoded = hf.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        if torch is not None:
            encoded = {k: v.to(hf.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = hf.model(**encoded).logits
            pred_ids = logits.argmax(-1)[0].detach().cpu().tolist()
        else:  # pragma: no cover
            pred_ids = [0 for _ in offsets]

        labels = []
        for (start, end), pred_id in zip(offsets, pred_ids):
            if start == 0 and end == 0:
                labels.append(-100)
            else:
                labels.append(pred_id)
        if labels:
            labels[0] = -100

        id2label = hf.model.config.id2label
        spans = self._bio_to_spans(labels, offsets, id2label, text)
        spans = self._postprocess(text, spans)
        return remove_overlaps_prefer_longest(drop_invalid_spans(text, spans))

    def _bio_to_spans(
        self,
        label_ids: Sequence[int],
        offsets: Sequence[Sequence[int]],
        id2label: dict,
        text: Optional[str] = None,
    ) -> List[SpanTuple]:
        spans: List[SpanTuple] = []
        current_label: Optional[str] = None
        current_start: Optional[int] = None
        current_end: Optional[int] = None

        for label_id, (tok_start, tok_end) in zip(label_ids, offsets):
            if label_id == -100:
                if current_label is not None and current_start is not None and current_end is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = None
                current_start = None
                current_end = None
                continue

            label = id2label[int(label_id)]
            if label.startswith("B-"):
                if current_label is not None and current_start is not None and current_end is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = label[2:]
                current_start = tok_start
                current_end = tok_end
            elif label.startswith("I-") and current_label == label[2:]:
                current_end = tok_end
            else:
                if current_label is not None and current_start is not None and current_end is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = None
                current_start = None
                current_end = None

        if current_label is not None and current_start is not None and current_end is not None:
            spans.append((current_start, current_end, current_label))

        if text is None:
            return spans

        cleaned: List[SpanTuple] = []
        for start, end, label in spans:
            while start < end and text[start] in STRIP_LEADING:
                start += 1
            while end > start and text[end - 1] in STRIP_TRAILING:
                end -= 1
            if end > start:
                cleaned.append((start, end, label))
        return cleaned

    def _postprocess(self, text: str, spans: Iterable[SpanTuple]) -> List[SpanTuple]:
        final: List[SpanTuple] = []
        for start, end, label in spans:
            if label == "Email":
                snippet = text[start:end].lower()
                if snippet.startswith(("info@", "support@", "noreply@", "test@")):
                    continue
            final.append((start, end, label))
        return final
