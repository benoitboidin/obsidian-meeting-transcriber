"""Torch-free port of pyannote.audio's `speaker-diarization-community-1` pipeline.

Same models (pyannote segmentation-3.0, WeSpeaker ResNet34-LM as fine-tuned/packaged by
pyannote, community-1 PLDA) and the same algorithm (sliding-window powerset segmentation,
masked-statistics-pooling embeddings excluding overlap, AHC + VBx clustering, constrained
assignment, speaker counting and reconstruction), run with onnxruntime + numpy + scipy.

    turns, exclusive, centroids = diarize(audio_f32_16k)

Models (in MODEL_DIR):
    segmentation-3.0.onnx               x (N,1,160000) -> powerset log-probs (N,589,7)
    wespeaker-resnet34-masked.onnx      fbank (B,T,80), weights (B,S,T') -> (B,S,256)
    plda.npz, xvec_transform.npz        community-1 PLDA
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import onnxruntime as ort
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, softmax

MODEL_DIR = os.environ.get("MT_DIAR_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "diarization-v1"))
SEG_MODEL = "segmentation-3.0.onnx"
EMB_MODEL = "wespeaker-resnet34-masked.onnx"

SR = 16000
CHUNK = 10.0                    # segmentation window (s)
STEP = 0.1 * CHUNK              # community-1: segmentation_step = 0.1
WIN = int(CHUNK * SR)           # 160000 samples
# segmentation-3.0 output frames (pyannote Model.receptive_field)
FRAME_START, FRAME_DUR, FRAME_STEP = 0.0, 991 / SR, 270 / SR
EMB_MIN_SAMPLES = 400           # PretrainedSpeakerEmbedding.min_num_samples for WeSpeaker
# community-1 hyper-parameters (config.yaml)
AHC_THRESHOLD, FA, FB = 0.6, 0.07, 0.8
MIN_ACTIVE_RATIO = 0.2
LDA_DIM = 128
SEG_BATCH = 32
EMB_BATCH = 16                  # chunks per batch (x3 local speakers)


# ----------------------------------------------------------------------------- utils

def _session(path: str, providers: Optional[list[str]] = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # CPU: validated against pyannote, small install, and leaves the GPU to Whisper.
    return ort.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])


def _powerset_mapping(num_classes: int = 3, max_set_size: int = 2) -> np.ndarray:
    sets = [c for k in range(max_set_size + 1) for c in combinations(range(num_classes), k)]
    m = np.zeros((len(sets), num_classes), dtype=np.float32)
    for i, s in enumerate(sets):
        m[i, list(s)] = 1
    return m


def _closest_frame(t: float) -> int:
    """pyannote SlidingWindow(start=0, duration=FRAME_DUR, step=FRAME_STEP).closest_frame."""
    return int(np.rint((t - 0.0 - 0.5 * FRAME_DUR) / FRAME_STEP))


def _aggregate(scores: np.ndarray, hamming: bool = False, missing: float = np.nan,
               skip_average: bool = False, epsilon: float = 1e-12) -> np.ndarray:
    """pyannote Inference.aggregate for chunks starting at c*STEP. scores: (C, F, K) with NaNs."""
    num_chunks, num_frames_per_chunk, num_classes = scores.shape
    win = (np.hamming(num_frames_per_chunk) if hamming else np.ones(num_frames_per_chunk)).reshape(-1, 1)
    num_frames = _closest_frame(CHUNK + (num_chunks - 1) * STEP + 0.5 * FRAME_DUR) + 1
    out = np.zeros((num_frames, num_classes), dtype=np.float32)
    cnt = np.zeros((num_frames, num_classes), dtype=np.float32)
    msk = np.zeros((num_frames, num_classes), dtype=np.float32)
    for c in range(num_chunks):
        score = scores[c]
        mask = 1 - np.isnan(score)
        score = np.nan_to_num(score, nan=0.0)
        s = _closest_frame(c * STEP + 0.5 * FRAME_DUR)
        out[s:s + num_frames_per_chunk] += score * mask * win
        cnt[s:s + num_frames_per_chunk] += mask * win
        msk[s:s + num_frames_per_chunk] = np.maximum(msk[s:s + num_frames_per_chunk], mask)
    avg = out if skip_average else out / np.maximum(cnt, epsilon)
    avg[msk == 0.0] = missing
    return avg


# ----------------------------------------------------------------------------- fbank

def _mel_banks(num_bins: int = 80, nfft: int = 512, low: float = 20.0, high: float = SR / 2) -> np.ndarray:
    """torchaudio.compliance.kaldi.get_mel_banks (vtln off), padded to nfft//2+1 columns."""
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)  # noqa: E731
    num_fft_bins = nfft // 2
    fft_bin_width = SR / nfft
    mel_low, mel_high = mel(low), mel(high)
    delta = (mel_high - mel_low) / (num_bins + 1)
    b = np.arange(num_bins, dtype=np.float64)[:, None]
    left, center, right = mel_low + b * delta, mel_low + (b + 1) * delta, mel_low + (b + 2) * delta
    m = mel(fft_bin_width * np.arange(num_fft_bins))[None, :]
    up = (m - left) / (center - left)
    down = (right - m) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)   # (80, 257)


_MEL = _mel_banks()
_HAMMING = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 399)).astype(np.float32)


def fbank(waveforms: np.ndarray) -> np.ndarray:
    """pyannote WeSpeaker compute_fbank: kaldi fbank (80 mel, 25/10 ms, hamming, dither 0,
    snip_edges, power spectrum, log) on samples scaled to int16 range, then mean normalisation.
    waveforms: (B, N) float32 in [-1, 1] -> (B, T, 80)."""
    x = waveforms.astype(np.float32) * 32768.0
    B, N = x.shape
    T = 1 + (N - 400) // 160
    idx = np.arange(400)[None, :] + 160 * np.arange(T)[:, None]
    fr = x[:, idx]                                               # (B, T, 400)
    fr = fr - fr.mean(axis=2, keepdims=True)                     # remove DC offset
    prev = np.concatenate([fr[:, :, :1], fr[:, :, :-1]], axis=2)
    fr = fr - 0.97 * prev                                        # pre-emphasis
    fr = fr * _HAMMING
    spec = np.abs(np.fft.rfft(fr, n=512, axis=2)) ** 2           # (B, T, 257)
    mel = spec.astype(np.float32) @ _MEL.T
    feats = np.log(np.maximum(mel, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=1, keepdims=True)).astype(np.float32)


# ----------------------------------------------------------------------------- PLDA / VBx

class PLDA:
    def __init__(self, transform_npz: str, plda_npz: str, lda_dim: int = LDA_DIM):
        x = np.load(transform_npz)
        self.mean1, self.mean2, self.lda = x["mean1"], x["mean2"], x["lda"]
        p = np.load(plda_npz)
        self.mu, tr, psi = p["mu"], p["tr"], p["psi"]
        W = np.linalg.inv(tr.T.dot(tr))
        Bm = np.linalg.inv((tr.T / psi).dot(tr))
        acvar, wccn = eigh(Bm, W)
        self.psi = acvar[::-1]
        self.tr = wccn.T[::-1]
        self.lda_dim = lda_dim

    @property
    def phi(self) -> np.ndarray:
        return self.psi[: self.lda_dim]

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=1, ord=2)[:, None]

    def __call__(self, e: np.ndarray) -> np.ndarray:
        lda = self.lda
        x = np.sqrt(lda.shape[1]) * self._l2(lda.T.dot(np.sqrt(lda.shape[0]) * self._l2(e - self.mean1).T).T - self.mean2)
        return (x - self.mu).dot(self.tr.T)[:, : self.lda_dim]


def _vbx(X, Phi, Fa, Fb, gamma, maxIters=20, epsilon=1e-4):
    """pyannote.audio.utils.vbx.VBx with given initial gamma (pi = number of init clusters)."""
    pi = np.ones(gamma.shape[1]) / gamma.shape[1]
    D = X.shape[1]
    G = -0.5 * (np.sum(X ** 2, axis=1, keepdims=True) + D * np.log(2 * np.pi))
    V = np.sqrt(Phi)
    rho = X * V
    prev = None
    for ii in range(maxIters):
        invL = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi)
        alpha = Fa / Fb * invL * gamma.T.dot(rho)
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha ** 2).dot(Phi) + G)
        lpi = np.log(pi + 1e-8)
        log_p_x = logsumexp(log_p_ + lpi, axis=-1)
        gamma = np.exp(log_p_ + lpi - log_p_x[:, None])
        pi = np.sum(gamma, axis=0)
        pi = pi / pi.sum()
        elbo = np.sum(log_p_x, axis=0) + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha ** 2 + 1)
        if ii > 0 and elbo - prev < epsilon:
            break
        prev = elbo
    return gamma, pi


def _kmeans(x: np.ndarray, k: int, n_init: int = 3, seed: int = 42, iters: int = 300) -> np.ndarray:
    """k-means++ / Lloyd (stands in for sklearn KMeans(n_init=3, random_state=42))."""
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        c = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d = np.min(cdist(x, np.array(c), "sqeuclidean"), axis=1)
            c.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(cdist(x, c, "sqeuclidean"), axis=1)
            new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        lab = np.argmin(cdist(x, c, "sqeuclidean"), axis=1)
        inertia = np.sum((x - c[lab]) ** 2)
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


# ----------------------------------------------------------------------------- pipeline

@dataclass
class Models:
    seg: ort.InferenceSession
    emb: ort.InferenceSession
    plda: PLDA


_models: Optional[Models] = None


def load_models(model_dir: str = MODEL_DIR, providers: Optional[list[str]] = None) -> Models:
    global _models
    if _models is None:
        _models = Models(
            seg=_session(os.path.join(model_dir, SEG_MODEL), providers),
            emb=_session(os.path.join(model_dir, EMB_MODEL), providers),
            plda=PLDA(os.path.join(model_dir, "xvec_transform.npz"), os.path.join(model_dir, "plda.npz")),
        )
    return _models


def _chunks(audio: np.ndarray) -> tuple[int, int]:
    """(number of full chunks, has_last_chunk) as in pyannote Inference.slide."""
    step = round(STEP * SR)
    n = len(audio)
    full = (n - WIN) // step + 1 if n >= WIN else 0
    has_last = n < WIN or (n - WIN) % step > 0
    return full, int(has_last)


def _chunk(audio: np.ndarray, c: int) -> np.ndarray:
    s = c * round(STEP * SR)
    x = audio[s:s + WIN]
    return np.pad(x, (0, WIN - len(x))) if len(x) < WIN else x


def segment(m: Models, audio: np.ndarray, progress=None) -> np.ndarray:
    """Binarized multilabel segmentation (C, 589, 3)."""
    full, last = _chunks(audio)
    num_chunks = full + last
    mapping = _powerset_mapping()
    out = []
    for b in range(0, num_chunks, SEG_BATCH):
        batch = np.stack([_chunk(audio, c) for c in range(b, min(b + SEG_BATCH, num_chunks))])[:, None, :]
        y = m.seg.run(None, {"x": batch.astype(np.float32)})[0]
        onehot = np.eye(y.shape[-1], dtype=np.float32)[np.argmax(y, axis=-1)]
        out.append(onehot @ mapping)
        if progress:
            progress("segmentation", min(1.0, (b + SEG_BATCH) / num_chunks))
    return np.concatenate(out)


def embed(m: Models, audio: np.ndarray, seg: np.ndarray, progress=None) -> np.ndarray:
    """(C, 3, 256) embeddings (exclude_overlap=True). Like pyannote, silent local speakers still
    get a (meaningless) embedding; clustering ignores them."""
    num_chunks, num_frames, num_spk = seg.shape
    min_num_frames = int(np.ceil(num_frames * EMB_MIN_SAMPLES / WIN))
    clean = seg * (np.sum(seg, axis=2, keepdims=True) < 2)
    use_clean = np.sum(clean, axis=1) > min_num_frames               # (C, S)
    masks = np.where(use_clean[:, None, :], clean, seg)               # (C, F, S)
    out = np.zeros((num_chunks, num_spk, 256), dtype=np.float32)
    for b in range(0, num_chunks, EMB_BATCH):
        cs = range(b, min(b + EMB_BATCH, num_chunks))
        fb = fbank(np.stack([_chunk(audio, c) for c in cs]))
        # weights interpolated ("nearest") to the ResNet output resolution
        tp = (fb.shape[1] + 7) // 8                                    # stride 8 in time
        idx = np.floor(np.arange(tp) * (num_frames / tp)).astype(int)
        w = np.transpose(masks[b:b + len(cs)], (0, 2, 1))[:, :, idx].astype(np.float32)
        e = m.emb.run(None, {"fbank": fb, "weights": w})[0]
        out[b:b + len(cs)] = e
        if progress:
            progress("embeddings", min(1.0, (b + EMB_BATCH) / num_chunks))
    return out


def cluster(m: Models, emb: np.ndarray, seg: np.ndarray, num_clusters=None, min_clusters=None,
            max_clusters=None) -> tuple[np.ndarray, np.ndarray]:
    """VBxClustering (constrained assignment). Returns hard_clusters (C,S), centroids (K,256)."""
    num_chunks, num_spk, dim = emb.shape
    single = np.sum(seg, axis=2, keepdims=True) == 1
    active = np.sum(seg * single, axis=1) >= MIN_ACTIVE_RATIO * seg.shape[1]
    valid = ~np.any(np.isnan(emb), axis=2)
    ci, si = np.where(active * valid)
    train = emb[ci, si]
    n = len(train)
    # set_num_clusters
    min_c = max(1, min(n, num_clusters or min_clusters or 1))
    max_c = max(1, min(n, num_clusters or max_clusters or n))
    if min_c == max_c:
        num_clusters = min_c
    if n < 2:
        return np.zeros((num_chunks, num_spk), dtype=np.int8), np.mean(train, axis=0, keepdims=True)

    normed = train / np.linalg.norm(train, axis=1, keepdims=True)
    ahc = fcluster(linkage(normed, method="centroid", metric="euclidean"), AHC_THRESHOLD, criterion="distance") - 1
    _, ahc = np.unique(ahc, return_inverse=True)
    qinit = np.zeros((len(ahc), ahc.max() + 1))
    qinit[range(len(ahc)), ahc] = 1.0
    q, sp = _vbx(m.plda(train), m.plda.phi, FA, FB, softmax(qinit * 7.0, axis=1))
    W = q[:, sp > 1e-7]
    centroids = W.T @ train / W.sum(0, keepdims=True).T

    constrained = True
    auto = centroids.shape[0]
    if auto < min_c:
        num_clusters = min_c
    elif auto > max_c:
        num_clusters = max_c
    if num_clusters and num_clusters != auto:
        constrained = False
        km = _kmeans(normed, num_clusters)
        centroids = np.vstack([np.mean(train[km == k], axis=0) for k in range(num_clusters)])

    with np.errstate(invalid="ignore"):
        soft = 2 - cdist(emb.reshape(-1, dim), centroids, metric="cosine").reshape(num_chunks, num_spk, -1)
    if constrained:
        soft[seg.sum(1) == 0] = soft.min() - 1.0
        soft = np.nan_to_num(soft, nan=np.nanmin(soft))
        hard = -2 * np.ones((num_chunks, num_spk), dtype=np.int8)
        for c, cost in enumerate(soft):
            for s, k in zip(*linear_sum_assignment(cost, maximize=True)):
                hard[c, s] = k
    else:
        hard = np.argmax(soft, axis=2)
    return hard, centroids


def _to_turns(binary: np.ndarray) -> list[tuple[float, float, int]]:
    """pyannote Binarize(onset=offset=0.5, min_duration_off=0) on frame middles."""
    t = FRAME_START + np.arange(binary.shape[0]) * FRAME_STEP + FRAME_DUR / 2
    turns = []
    for k in range(binary.shape[1]):
        a = binary[:, k] > 0.5
        if not a.any():
            continue
        d = np.diff(a.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if a[0]:
            starts.insert(0, 0)
        for i, s in enumerate(starts):
            e_idx = ends[i] if i < len(ends) else len(a) - 1
            if t[e_idx] > t[s]:
                turns.append((float(t[s]), float(t[e_idx]), k))
    turns.sort()
    return turns


def _reconstruct(seg: np.ndarray, hard: np.ndarray, count: np.ndarray) -> np.ndarray:
    num_chunks, num_frames, _ = seg.shape
    K = int(np.max(hard)) + 1
    cs = np.full((num_chunks, num_frames, K), np.nan)
    for c in range(num_chunks):
        for k in np.unique(hard[c]):
            if k == -2:
                continue
            cs[c, :, k] = np.max(seg[c][:, hard[c] == k], axis=1)
    act = _aggregate(cs, missing=0.0, skip_average=True)
    maxc = int(count.max())
    if act.shape[1] < maxc:
        act = np.pad(act, ((0, 0), (0, maxc - act.shape[1])))
    n = min(len(act), len(count))
    act, cnt = act[:n], count[:n, 0]
    order = np.argsort(-act, axis=-1)
    binary = np.zeros_like(act)
    for t in range(n):
        binary[t, order[t, :cnt[t]]] = 1.0
    return binary


def diarize(audio: np.ndarray, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None,
            num_speakers: Optional[int] = None, progress=None, models: Optional[Models] = None):
    """audio: float32 mono 16 kHz in [-1, 1].
    Returns (turns, exclusive_turns, centroids): turns are (start, end, "SPEAKER_xx");
    centroids maps each label to its 256-d embedding (same space as pyannote's speaker_embeddings)."""
    m = models or load_models()
    audio = np.asarray(audio, dtype=np.float32)
    # set_num_speakers
    min_speakers = num_speakers or min_speakers or 1
    max_speakers = num_speakers or max_speakers or np.inf
    if min_speakers == max_speakers:
        num_speakers = min_speakers

    seg = segment(m, audio, progress)
    count = np.rint(_aggregate(np.sum(seg, axis=-1, keepdims=True), missing=0.0)).astype(np.uint8)
    if np.nanmax(count) == 0:
        return [], [], {}
    emb = embed(m, audio, seg, progress)
    hard, centroids = cluster(m, emb, seg, num_speakers, min_speakers,
                              None if max_speakers == np.inf else max_speakers)
    count = np.minimum(count, max_speakers).astype(np.int8)
    hard = hard.astype(np.int16)
    hard[np.sum(seg, axis=1) == 0] = -2

    binary = _reconstruct(seg, hard, count)
    excl = _reconstruct(seg, hard, np.minimum(count, 1).astype(np.int8))
    turns, ex_turns = _to_turns(binary), _to_turns(excl)

    labels = sorted({k for _, _, k in turns})
    if len(labels) and max(labels) >= centroids.shape[0]:
        centroids = np.pad(centroids, ((0, max(labels) + 1 - centroids.shape[0]), (0, 0)))
    name = {k: f"SPEAKER_{i:02d}" for i, k in enumerate(labels)}
    out = [(s, e, name[k]) for s, e, k in turns]
    ex = [(s, e, name[k]) for s, e, k in ex_turns if k in name]
    return out, ex, {name[k]: centroids[k] for k in labels}
