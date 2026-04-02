"""
training/evaluator.py
======================
Inference-time evaluation utilities for the Tiny-Shakespeare Transformer.

Provides:
    compute_accuracy   — teacher-forced token-level accuracy on any DataLoader
    generate_text      — autoregressive text generation with temperature / top-k
    run_demo_generation — high-level wrapper that prints multiple generation examples
    model_summary      — prints parameter counts per sub-network

Text generation uses the encoder-decoder design:
    encoder input  ← prompt (character sequence)
    decoder output ← continuation (generated character by character)

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.model.transformer import Transformer
from data.dataset import CharTokenizer


# ── Token-level accuracy ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(
    model:        Transformer,
    loader:       DataLoader,
    device:       torch.device,
    pad_token_id: int = 0,
) -> float:
    """
    Compute teacher-forced token-level accuracy.

    Runs the model in eval mode using the ground-truth decoder input (trg)
    and counts how many predicted tokens match the target labels, excluding
    <PAD> positions.

    Args:
        model        : Trained Transformer (set to eval mode internally).
        loader       : DataLoader yielding (src, trg, labels).
        device       : Compute device.
        pad_token_id : <PAD> ID — excluded from accuracy calculation.

    Returns:
        accuracy : Fraction of correct non-PAD token predictions.
    """
    model.eval()
    correct: int = 0
    total:   int = 0

    for src, trg, labels in loader:
        src    = src.to(device,    non_blocking=True)
        trg    = trg.to(device,    non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits: torch.Tensor = model(src, trg)        # (B, T, V)
        preds:  torch.Tensor = logits.argmax(dim=-1)  # (B, T)

        non_pad: torch.Tensor = labels != pad_token_id
        correct += int((preds[non_pad] == labels[non_pad]).sum().item())
        total   += int(non_pad.sum().item())

    return correct / max(total, 1)


# ── Autoregressive text generation ───────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model:       Transformer,
    tokenizer:   CharTokenizer,
    prompt:      str,
    max_gen_len: int   = 200,
    device:      Optional[torch.device] = None,
    temperature: float = 1.0,
    top_k:       int   = 0,
) -> str:
    """
    Generate Shakespeare-like text autoregressively from a prompt.

    The prompt is fed as the encoder source.  The decoder generates
    tokens one at a time by sampling (or greedy decoding) at each step.

    Decoding modes:
        temperature=1.0, top_k=0  → greedy (argmax)
        temperature < 1.0         → sharper distribution (less creative)
        temperature > 1.0         → flatter distribution (more creative)
        top_k > 0                 → restrict sampling to top-k logits

    Args:
        model       : Trained Transformer.
        tokenizer   : CharTokenizer for encoding / decoding.
        prompt      : Seed string (encoder input).
        max_gen_len : Maximum characters to generate.
        device      : Compute device (inferred from model parameters if None).
        temperature : Softmax temperature.
        top_k       : If > 0, limit sampling to the top-k most likely tokens.

    Returns:
        Full string: prompt + generated continuation.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Encode prompt → source tensor
    src_ids: List[int] = tokenizer.encode(prompt)
    src: torch.Tensor = torch.tensor(
        src_ids, dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, T_src)

    # Pre-compute encoder output once
    src_mask: torch.Tensor  = model.make_src_mask(src)
    encoder_out, _ = model.encoder(src, src_mask)

    # Initialise decoder sequence with <SOS>
    trg: torch.Tensor = torch.tensor(
        [[tokenizer.sos_id]], dtype=torch.long, device=device
    )
    generated_ids: List[int] = []
    max_len = 256

    for _ in range(max_gen_len):
            
            # Crop the target sequence if it exceeds the model's maximum length capacity
            if trg.size(1) > max_len:
                trg_input = trg[:, -max_len:]
            else:
                trg_input = trg    
            # Use the cropped `trg_input` for the forward pass, NOT the full `trg`
            trg_mask: torch.Tensor = model.make_trg_mask(trg_input)
            dec_out, _, _ = model.decoder(
                trg_input, encoder_out, trg_mask=trg_mask, src_mask=src_mask
            )
    
            # Logits for the last generated position
            logits: torch.Tensor = model.output_proj(dec_out[:, -1, :])  # (1, V)
    
            if top_k > 0 or temperature != 1.0:
                # Apply temperature scaling
                logits = logits / max(temperature, 1e-8)
    
                if top_k > 0:
                    # Zero out all logits below the k-th highest value
                    k = min(top_k, logits.size(-1))
                    top_vals, _ = torch.topk(logits, k)
                    threshold: torch.Tensor = top_vals[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
    
                probs: torch.Tensor = F.softmax(logits, dim=-1)
                next_token: torch.Tensor = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
    
            token_id: int = int(next_token.item())
    
            if token_id == tokenizer.eos_id:
                break
    
            generated_ids.append(token_id)
            
            trg = torch.cat([trg, next_token], dim=1)
    
    return prompt + tokenizer.decode(generated_ids)


# ── High-level demo ───────────────────────────────────────────────────────────

def run_demo_generation(
    model:       Transformer,
    tokenizer:   CharTokenizer,
    prompts:     Optional[List[str]] = None,
    max_gen_len: int   = 300,
    device:      Optional[torch.device] = None,
    temperature: float = 0.8,
    top_k:       int   = 10,
) -> None:
    """
    Run text generation from a list of Shakespeare-style prompts and print
    the results.  Useful as a qualitative check during and after training.

    Args:
        model       : Trained Transformer.
        tokenizer   : CharTokenizer.
        prompts     : Seed strings.  Uses canonical Shakespeare lines if None.
        max_gen_len : Maximum generation length (characters).
        device      : Compute device (inferred if None).
        temperature : Sampling temperature.
        top_k       : Top-k sampling pool size.
    """
    if prompts is None:
        prompts = [
            "ROMEO:",
            "To be, or not to be",
            "HAMLET:\nWhat a piece of work is man",
            "KING HENRY:",
        ]

    if device is None:
        device = next(model.parameters()).device

    sep = "=" * 60
    print(f"\n{sep}")
    print("  TEXT GENERATION DEMO")
    print(f"  temperature={temperature}  top_k={top_k}")
    print(sep)

    for prompt in prompts:
        generated: str = generate_text(
            model       = model,
            tokenizer   = tokenizer,
            prompt      = prompt,
            max_gen_len = max_gen_len,
            device      = device,
            temperature = temperature,
            top_k       = top_k,
        )
        print(f"\n[Prompt] {prompt!r}")
        print("[Output]")
        print(generated)
        print("-" * 40)

    print()


# ── Model inspection ──────────────────────────────────────────────────────────

def model_summary(model: Transformer) -> None:
    """Print a concise overview of the model's parameter counts."""

    def count(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    enc_p  = count(model.encoder)
    dec_p  = count(model.decoder)
    proj_p = count(model.output_proj)
    total  = enc_p + dec_p + proj_p

    print("\n" + "=" * 42)
    print("  Transformer Parameter Summary")
    print("=" * 42)
    print(f"  Encoder            : {enc_p:>10,}")
    print(f"  Decoder            : {dec_p:>10,}")
    print(f"  Output Projection  : {proj_p:>10,}")
    print(f"  {'─' * 30}")
    print(f"  Total trainable    : {total:>10,}")
    print("=" * 42 + "\n")

