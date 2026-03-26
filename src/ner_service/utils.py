from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


SpanTuple = Tuple[int, int, str]


def sort_spans(spans: Iterable[SpanTuple]) -> List[SpanTuple]:
    return sorted(spans, key=lambda item: (item[0], item[1], item[2]))


def drop_invalid_spans(text: str, spans: Sequence[SpanTuple]) -> List[SpanTuple]:
    valid: List[SpanTuple] = []
    for start, end, label in spans:
        if 0 <= start < end <= len(text):
            valid.append((start, end, label))
    return valid


def remove_overlaps_prefer_longest(spans: Sequence[SpanTuple]) -> List[SpanTuple]:
    ordered = sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0]))
    kept: List[SpanTuple] = []
    for span in ordered:
        start, end, _ = span
        overlaps = any(not (end <= ks or start >= ke) for ks, ke, _ in kept)
        if not overlaps:
            kept.append(span)
    return sort_spans(kept)


def label_to_placeholder_prefix(label: str) -> str:
    normalized = label.upper().replace(" ", "_").replace("-", "_")
    replacements = {
        "EMAIL": "EMAIL",
        "НОМЕР_ТЕЛЕФОНА": "PHONE",
        "PHONE": "PHONE",
        "ФИО": "PERSON",
        "ПАСПОРТНЫЕ_ДАННЫЕ": "PASSPORT",
        "СВЕДЕНИЯ_ОБ_ИНН": "INN",
        "СНИЛС_КЛИЕНТА": "SNILS",
        "НОМЕР_КАРТЫ": "CARD",
        "НОМЕР_БАНКОВСКОГО_СЧЕТА": "ACCOUNT",
        "ПОЛНЫЙ_АДРЕС": "ADDRESS",
        "НАИМЕНОВАНИЕ_БАНКА": "BANK",
        "CVV/CVC": "CVV",
    }
    return replacements.get(normalized, normalized)
