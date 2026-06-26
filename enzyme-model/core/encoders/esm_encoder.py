"""
Shared ESM Encoder

Consolidates seq_to_vec implementations from:
  - predict_kcat.py (lines 140-200)
  - predict_km.py (lines 328-408)
  - train/train_kcat.py (lines 209-334)
  - train/train_km.py (lines 210-336)

Usage:
    from core.encoders.esm_encoder import ESMEncoder, encode_sequences_dual_gpu

    # Standard inference (predict)
    encoder = ESMEncoder()
    embedding = encoder.encode("MKLL...")

    # Dual-GPU training parallel
    features = encode_sequences_dual_gpu(sequences)
"""

import os
import gc
import torch
import numpy as np


def _check_esm_available():
    """Returns True if the esm package can be imported."""
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        return True
    except ImportError:
        return False


ESM_AVAILABLE = _check_esm_available()

if ESM_AVAILABLE:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig


# ═══════════════════════════════════════════════════════════════════
# Single-model encoder class  (used by predict scripts)
# ═══════════════════════════════════════════════════════════════════

class ESMEncoder:
    """Unified protein sequence → embedding vector encoder.

    Wraps an ESMC-300M model and exposes ``encode()`` that optionally
    batches sequences on GPU with automatic CPU fallback.

    Parameters
    ----------
    model_name : str
        HuggingFace identifier for the ESM model (default ``"esmc_300m"``).
    embed_dim : int
        Expected embedding dimension (default ``960``).
    device : torch.device or None
        Target device.  ``None`` → auto-detect with ``get_device()``.
    use_direct_load : bool
        * ``True``  → ``ESMC.from_pretrained(name, device=device)``
        * ``False`` → ``ESMC.from_pretrained(name).to(device)``

        This preserves the exact loading API used by the original call site
        so that numerical results are identical.
    """

    def __init__(self, model_name="esmc_300m", embed_dim=960,
                 device=None, use_direct_load=True):
        if not ESM_AVAILABLE:
            raise RuntimeError("ESM library not installed – cannot create ESMEncoder")

        self.model_name = model_name
        self.embed_dim = embed_dim

        if device is None:
            device = self._get_device()

        os.environ["INFRA_PROVIDER"] = "True"
        try:
            if use_direct_load:
                # Matches predict_kcat.py: line 156
                self.client = ESMC.from_pretrained(model_name, device=device)
            else:
                # Matches predict_km.py: line 344 and train scripts
                self.client = ESMC.from_pretrained(model_name).to(device)
        except Exception as e:
            if device.type == 'cuda':
                # GPU loading failed – try CPU fallback (predict_km pattern)
                device = torch.device("cpu")
                os.environ["INFRA_PROVIDER"] = "True"
                if use_direct_load:
                    self.client = ESMC.from_pretrained(model_name, device=device)
                else:
                    self.client = ESMC.from_pretrained(model_name).to(device)
            else:
                raise RuntimeError(f"Cannot load ESM model '{model_name}': {e}")

        self.client.eval()
        self._device = device

    # ── public API ──────────────────────────────────────────────

    def encode(self, sequences, batch_size=None):
        """Encode one or more protein sequences into an (N, 960) numpy array.

        Parameters
        ----------
        sequences : str or list[str]
            Amino-acid sequence(s).
        batch_size : int or None
            Batch size for internal iteration.  ``None`` → process all at once.

        Returns
        -------
        np.ndarray
            Shape (1, embed_dim) for a single sequence, (N, embed_dim) for many.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        all_features = []

        with torch.no_grad():
            for seq in sequences:
                try:
                    protein = ESMProtein(sequence=seq)
                    protein_tensor = self.client.encode(protein)
                    logits_output = self.client.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_embeddings=True)
                    )
                    embeddings = logits_output.embeddings.squeeze(0).cpu().numpy()
                    avg_embedding = np.mean(embeddings, axis=0)
                except Exception:
                    avg_embedding = np.zeros(self.embed_dim, dtype=np.float32)

                all_features.append(avg_embedding)

        return np.array(all_features, dtype=np.float32)

    def release(self):
        """Free GPU memory held by this encoder."""
        del self.client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _get_device():
        """Detect a usable device (CPU-safe, memory-test on GPU)."""
        if torch.cuda.is_available():
            try:
                _ = torch.zeros((100, 100), device="cuda")
                return torch.device("cuda")
            except RuntimeError:
                pass
        return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════
# Training-specific helpers  (dual-GPU + single-GPU progress)
# ═══════════════════════════════════════════════════════════════════

def _process_chunk_gpu(chunk_sequences, device_id, model_name, embed_dim):
    """Process a chunk of sequences on a single GPU (for dual-GPU training).

    Matches train/train_kcat.py lines 209-240 and train/train_km.py.
    """
    import os as _os
    _os.environ["INFRA_PROVIDER"] = "True"
    device = torch.device(f"cuda:{device_id}")

    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()

    results = []
    with torch.no_grad():
        for seq in chunk_sequences:
            try:
                protein = ESMProtein(sequence=seq)
                protein_tensor = client.encode(protein)
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings = logits_output.embeddings.squeeze(0).cpu().numpy()
                results.append(np.mean(embeddings, axis=0))
            except Exception:
                results.append(np.zeros(embed_dim, dtype=np.float32))

    del client
    torch.cuda.empty_cache()
    return results


def _seq_to_vec_single_gpu(Sequence, device_id, model_name, embed_dim):
    """Encode all sequences on a single GPU with progress reporting.

    Matches train/train_kcat.py lines 243-275 and train/train_km.py.
    """
    device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()

    total = len(Sequence)
    all_features = []
    report_every = max(total // 20, 10)

    with torch.no_grad():
        for i, seq in enumerate(Sequence):
            try:
                protein = ESMProtein(sequence=seq)
                protein_tensor = client.encode(protein)
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings = logits_output.embeddings.squeeze(0).cpu().numpy()
                all_features.append(np.mean(embeddings, axis=0))
            except Exception:
                all_features.append(np.zeros(embed_dim, dtype=np.float32))

            if (i + 1) % report_every == 0 or (i + 1) == total:
                pct = int((i + 1) / total * 100)
                print(f"PROGRESS:{pct}", flush=True)

    del client
    torch.cuda.empty_cache()
    return np.array(all_features, dtype=np.float32)


def encode_sequences_dual_gpu(sequences, model_name="esmc_300m", embed_dim=960,
                              batch_size=200):
    """Encode protein sequences using dual-GPU parallelism (training use).

    Matches train/train_kcat.py lines 278-334 and train/train_km.py.

    Falls back to single-GPU if < 2 GPUs are available or one GPU fails.
    """
    import concurrent.futures
    num_gpus = torch.cuda.device_count()
    print(f"开始蛋白质特征提取: {model_name}")
    print(f"检测到 {num_gpus} 块 GPU")

    if num_gpus < 2:
        return _seq_to_vec_single_gpu(
            sequences, (0 if torch.cuda.is_available() else -1),
            model_name, embed_dim
        )

    chunk_size = (len(sequences) + 1) // 2
    chunks = [sequences[:chunk_size], sequences[chunk_size:]]

    all_features = [(0, None), (1, None)]
    gpu_ok = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for gpu_id, chunk in enumerate(chunks):
            if not chunk:
                continue
            future = executor.submit(_process_chunk_gpu, chunk, gpu_id, model_name, embed_dim)
            futures[future] = gpu_id

        for future in concurrent.futures.as_completed(futures):
            gpu_id = futures[future]
            try:
                result = future.result()
                if result is None:
                    gpu_ok = False
                else:
                    all_features[gpu_id] = (gpu_id, result)
            except Exception:
                gpu_ok = False

    if not gpu_ok:
        print("  ⚠️ 双 GPU 模式失败，回退到单 GPU（GPU 0）", flush=True)
        return _seq_to_vec_single_gpu(sequences, 0, model_name, embed_dim)

    all_features.sort(key=lambda x: x[0])
    merged = []
    for _, feat in all_features:
        if feat is not None:
            merged.extend(feat)
    return np.array(merged, dtype=np.float32)
