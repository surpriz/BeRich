"""FinBERT scoring on GPU for cached news rows.

Wraps ``ProsusAI/finbert`` (HuggingFace) behind a tiny interface: ``score_texts``
returns the ``(n, 3)`` softmax probabilities ``[negative, neutral, positive]``
for a batch of input strings. The model is single-load (lazy at first call) and
runs on CUDA when available, half-precision when supported, batch 64 with a
512-token max — long enough for our title + summary concatenation in practice.

The model is in ``eval`` mode and we pass ``torch.inference_mode``: no dropout,
no gradient state, deterministic outputs for identical inputs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
LABEL_ORDER = ("negative", "neutral", "positive")  # FinBERT's published convention
DEFAULT_BATCH = 64
MAX_TOKENS = 256  # title + first paragraph of summary fit well under 256 tokens


class FinBertScorer:
    """Single-load FinBERT scorer.

    Construction is cheap; the model + tokenizer are pulled on the first
    ``score_texts`` call so importing this module costs nothing (the scheduler
    imports a lot of things at startup).
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH,
        use_half_precision: bool | None = None,
    ) -> None:
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # half-precision only on CUDA; CPU runs FP32.
        self._use_half = (
            use_half_precision if use_half_precision is not None else self.device.type == "cuda"
        )
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        # Import lazily so the rest of the codebase doesn't pay for transformers
        # at module-import time (it's a heavy import; ~3s on cold cache).
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        logger.info("loading FinBERT (%s) on %s", MODEL_NAME, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(self.device)
        model.eval()
        if self._use_half:
            model = model.half()
        self._model = model

    def score_texts(self, texts: list[str]) -> np.ndarray:
        """Return ``(n, 3)`` softmax probabilities aligned to ``texts`` and ``LABEL_ORDER``."""
        if not texts:
            return np.zeros((0, 3), dtype=np.float32)
        self._ensure_loaded()
        assert self._tokenizer is not None  # noqa: S101 — _ensure_loaded just set it
        assert self._model is not None  # noqa: S101

        # FinBERT publishes its label order via id2label; we re-index to our
        # canonical (negative, neutral, positive) so callers don't have to care
        # about HF's storage order. The cast keeps ty quiet about the stubbed
        # ``id2label`` being a union with None.
        id2label_raw = self._model.config.id2label or {}
        id2label = {int(k): v.lower() for k, v in id2label_raw.items()}
        target_indices = [
            next(i for i, lbl in id2label.items() if lbl == name) for name in LABEL_ORDER
        ]

        outputs: list[np.ndarray] = []
        n = len(texts)
        for start in range(0, n, self.batch_size):
            batch = [_safe(t) for t in texts[start : start + self.batch_size]]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**enc).logits  # (b, 3)
                probs = torch.softmax(logits.float(), dim=-1)
            outputs.append(probs.cpu().numpy()[:, target_indices])
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _safe(text: str | None) -> str:
    """Treat None / blank text as a neutral input rather than crashing the tokenizer."""
    if text is None:
        return ""
    return str(text).strip() or ""


__all__ = ["LABEL_ORDER", "MAX_TOKENS", "MODEL_NAME", "FinBertScorer"]
