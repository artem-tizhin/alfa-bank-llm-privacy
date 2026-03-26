from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .schemas import EntitySpan, MaskMapping
from .utils import label_to_placeholder_prefix, sort_spans


SpanTuple = Tuple[int, int, str]


def mask_text(text: str, spans: Sequence[SpanTuple]) -> tuple[str, List[MaskMapping]]:
    spans = sort_spans(spans)
    parts: List[str] = []
    mapping: List[MaskMapping] = []
    last_idx = 0
    counters: Dict[str, int] = {}

    for start, end, label in spans:
        prefix = label_to_placeholder_prefix(label)
        number = counters.get(prefix, 0)
        counters[prefix] = number + 1
        placeholder = f"[{prefix}_{number}]"

        parts.append(text[last_idx:start])
        parts.append(placeholder)
        mapping.append(
            MaskMapping(
                placeholder=placeholder,
                original_text=text[start:end],
                label=label,
                start=start,
                end=end,
            )
        )
        last_idx = end

    parts.append(text[last_idx:])
    return "".join(parts), mapping


def unmask_text(text: str, mapping: Sequence[MaskMapping]) -> str:
    result = text
    for item in sorted(mapping, key=lambda x: len(x.placeholder), reverse=True):
        result = result.replace(item.placeholder, item.original_text)
    return result


def spans_to_entities(text: str, spans: Sequence[SpanTuple]) -> List[EntitySpan]:
    return [
        EntitySpan(start=start, end=end, label=label, text=text[start:end])
        for start, end, label in sort_spans(spans)
    ]
