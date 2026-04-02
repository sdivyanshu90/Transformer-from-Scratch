"""
data/dataset.py
================
Tiny Shakespeare — Character-Level Seq2Seq Dataset

Downloads (and caches) the Tiny Shakespeare corpus from Karpathy's repo,
tokenises it at the character level, and yields (src, trg, labels) triples
that frame the data as a seq2seq prediction task for an Encoder-Decoder
Transformer.

──────────────────────────────────────────────────────────────────────────────
Seq2Seq Framing (Continuation Task)
──────────────────────────────────────────────────────────────────────────────
To prevent the Decoder from "cheating" via Cross-Attention, we split a 
2*L window into two halves:

    Encoder Input  (src)    : Chunk A (tokens[0 : L])
    Decoder Input  (trg)    : <SOS> + Chunk B (tokens[L : 2L-1])
    Decoder Target (labels) : Chunk B (tokens[L : 2L])

The model must read Chunk A, and autoregressively generate Chunk B.
"""

import os
import urllib.request
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ── Constants ─────────────────────────────────────────────────────────────────

_DATA_URL: str = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master"
    "/data/tinyshakespeare/input.txt"
)
_CACHE_PATH: str = os.path.join("data", "tinyshakespeare.txt")

# Special-token name → ID mapping
SPECIAL_TOKENS: Dict[str, int] = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}
PAD_ID: int = 0
SOS_ID: int = 1
EOS_ID: int = 2
UNK_ID: int = 3
_SPECIAL_OFFSET: int = len(SPECIAL_TOKENS)   # = 4


# ── Character Tokenizer ───────────────────────────────────────────────────────

class CharTokenizer:
    """
    Character-level tokenizer for Tiny Shakespeare.
    Builds a vocabulary from the full corpus.
    """

    def __init__(self, text: str) -> None:
        # Sorted unique characters for deterministic vocab ordering
        unique_chars: List[str] = sorted(set(text))

        self.char_to_id: Dict[str, int] = {
            ch: idx + _SPECIAL_OFFSET for idx, ch in enumerate(unique_chars)
        }
        self.id_to_char: Dict[int, str] = {
            idx + _SPECIAL_OFFSET: ch for idx, ch in enumerate(unique_chars)
        }

        # Register special tokens in both lookup tables
        for token, token_id in SPECIAL_TOKENS.items():
            self.char_to_id[token] = token_id
            self.id_to_char[token_id] = token

        # Total vocabulary: special tokens + unique characters
        self.vocab_size: int = _SPECIAL_OFFSET + len(unique_chars)

        # Expose special-token IDs for convenience
        self.pad_id: int = PAD_ID
        self.sos_id: int = SOS_ID
        self.eos_id: int = EOS_ID
        self.unk_id: int = UNK_ID

    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of integer token IDs."""
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert a list of token IDs back to a string."""
        skip_set = {self.pad_id, self.sos_id, self.eos_id} if skip_special else set()
        return "".join(
            self.id_to_char.get(i, "<UNK>")
            for i in ids
            if i not in skip_set
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShakespeareDataset(Dataset):
    """
    Tiny Shakespeare Character-Level Seq2Seq Dataset.
    """

    def __init__(
        self,
        split:      str   = "train",
        chunk_size: int   = 128,
        train_frac: float = 0.9,
    ) -> None:
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.chunk_size: int = chunk_size

        # 1. Load (downloading if necessary)
        text: str = self._load_text()

        # 2. Build tokenizer from the full corpus
        self.tokenizer: CharTokenizer = CharTokenizer(text)

        # 3. Encode the entire corpus to a flat list of token IDs
        all_ids: List[int] = self.tokenizer.encode(text)

        # 4. Train / validation split (by token index)
        split_idx: int = int(len(all_ids) * train_frac)
        self._data: List[int] = (
            all_ids[:split_idx] if split == "train" else all_ids[split_idx:]
        )

        # 5. Pre-compute chunk starts
        # We now need 2 * chunk_size tokens per sequence (one chunk for encoder, one for decoder)
        n: int = len(self._data)
        self._starts: List[int] = list(range(0, n - (2 * chunk_size), chunk_size))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_text() -> str:
        """Download (once) and read the Tiny Shakespeare corpus."""
        os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
        if not os.path.exists(_CACHE_PATH):
            print(f"[dataset] Downloading Tiny Shakespeare → {_CACHE_PATH} …")
            urllib.request.urlretrieve(_DATA_URL, _CACHE_PATH)
            print("[dataset] Download complete.")
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            return fh.read()

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        start: int   = self._starts[idx]
        L: int = self.chunk_size

        # Source (Encoder Input): The first half of the data window
        src_chunk: List[int] = self._data[start : start + L]
        
        # Target (Decoder/Labels): The second half of the data window (the continuation)
        trg_chunk: List[int] = self._data[start + L : start + 2 * L]

        # Encoder input
        src: torch.Tensor = torch.tensor(src_chunk, dtype=torch.long)

        # Decoder input: <SOS> followed by all-but-last tokens of the target chunk
        trg: torch.Tensor = torch.tensor(
            [self.tokenizer.sos_id] + trg_chunk[: L - 1],
            dtype=torch.long,
        )

        # Labels: the full target chunk 
        labels: torch.Tensor = torch.tensor(trg_chunk, dtype=torch.long)

        return src, trg, labels


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    chunk_size:  int   = 128,
    batch_size:  int   = 64,
    train_frac:  float = 0.9,
    num_workers: int   = 0,
) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    """
    Build training and validation DataLoaders plus the shared tokenizer.
    """
    pin = torch.cuda.is_available()

    train_ds = ShakespeareDataset("train", chunk_size, train_frac)
    val_ds   = ShakespeareDataset("val",   chunk_size, train_frac)

    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # Both splits share the same tokenizer (built from the full corpus)
    return train_loader, val_loader, train_ds.tokenizer
