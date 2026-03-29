"""Post-hoc temperature scaling calibration for the chess GNN value head.

Temperature scaling learns a single scalar T such that
    calibrated_prob = sigmoid(logit / T)
matches the empirical Stockfish win-probability distribution on a held-out
calibration set.  T > 1 softens overconfident predictions; T < 1 sharpens
underconfident ones.

Usage
-----
scaler = TemperatureScaler()
scaler.fit(logits, targets)           # numpy arrays; targets are soft labels in [0, 1]
scaler.save("output/gateau_distilled.pt.calib.json")

# Later, during inference:
scaler = TemperatureScaler()
scaler.load("output/gateau_distilled.pt.calib.json")
calibrated_win_prob = scaler.calibrate(raw_win_prob)   # raw_win_prob in [0, 1]
"""

import json
import logging
import math

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_EPSILON = 1e-7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _prob_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
    return np.log(p / (1.0 - p))


def _nll_and_grad(log_t: float, logits: np.ndarray, targets: np.ndarray):
    """NLL of scaled logits vs soft targets, plus its gradient w.r.t. log(T)."""
    T = math.exp(log_t)
    scaled = logits / T
    probs = _sigmoid(scaled)
    probs = np.clip(probs, _EPSILON, 1.0 - _EPSILON)
    nll = -np.mean(
        targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs)
    )
    # d(NLL)/d(log T) = -mean((p - t) * s) = mean((t - p) * s)
    grad_scalar = np.mean((targets - probs) * scaled)
    return float(nll), np.array([grad_scalar])


class TemperatureScaler:
    """Post-hoc calibration via temperature scaling.

    Parameters
    ----------
    T : float
        Initial temperature (1.0 = no change).
    """

    def __init__(self, T: float = 1.0) -> None:
        self.T = T

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> None:
        """Fit T by minimising soft NLL on a calibration set.

        Parameters
        ----------
        logits : np.ndarray, shape (N,)
            Raw logits (``log-odds``) corresponding to each position's value
            head output.  Compute via ``_prob_to_logit(win_prob_array)``.
        targets : np.ndarray, shape (N,)
            Soft calibration targets in [0, 1].  Use Stockfish ``eval_wp``.
        """
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        result = minimize(
            fun=lambda lt: _nll_and_grad(lt[0], logits, targets),
            x0=[0.0],  # log(T) = 0  →  T = 1
            jac=True,
            method="L-BFGS-B",
            bounds=[(-3.0, 3.0)],  # T ∈ [0.05, 20]
            options={"maxiter": 500, "ftol": 1e-12},
        )
        self.T = float(math.exp(result.x[0]))
        logger.info("TemperatureScaler fitted: T=%.4f (converged=%s)", self.T, result.success)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(self, win_prob: float) -> float:
        """Apply temperature scaling to a single win probability.

        Parameters
        ----------
        win_prob : float
            Raw model win probability in [0, 1].

        Returns
        -------
        float
            Calibrated win probability in [0, 1].
        """
        p = float(np.clip(win_prob, _EPSILON, 1.0 - _EPSILON))
        logit = math.log(p / (1.0 - p))
        scaled_logit = logit / self.T
        return float(1.0 / (1.0 + math.exp(-scaled_logit)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save temperature to a JSON sidecar file."""
        with open(path, "w") as fh:
            json.dump({"temperature": self.T}, fh, indent=2)
        logger.info("TemperatureScaler saved to %s (T=%.4f)", path, self.T)

    def load(self, path: str) -> None:
        """Load temperature from a JSON sidecar file."""
        with open(path) as fh:
            data = json.load(fh)
        self.T = float(data["temperature"])
        logger.info("TemperatureScaler loaded from %s (T=%.4f)", path, self.T)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def ece(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Parameters
        ----------
        probs : np.ndarray, shape (N,)
            Predicted win probabilities in [0, 1].
        targets : np.ndarray, shape (N,)
            Ground-truth targets in [0, 1].
        n_bins : int
            Number of equal-width bins.

        Returns
        -------
        float
            ECE ∈ [0, 1]; lower is better.
        """
        probs = np.asarray(probs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_val = 0.0
        N = len(probs)
        if N == 0:
            return 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if not mask.any():
                continue
            bin_probs = probs[mask]
            bin_targets = targets[mask]
            ece_val += (mask.sum() / N) * abs(bin_probs.mean() - bin_targets.mean())
        return float(ece_val)


# ---------------------------------------------------------------------------
# Standalone reliability diagram (array-based; complements eval.py's JSONL version)
# ---------------------------------------------------------------------------


def reliability_diagram(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
):
    """Build a reliability diagram from pre-computed probability arrays.

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Predicted win probabilities in [0, 1].
    targets : np.ndarray, shape (N,)
        Ground-truth targets in [0, 1].
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    matplotlib.figure.Figure
        The reliability diagram figure.
    """
    import matplotlib.pyplot as plt

    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    mean_preds: list[float] = []
    mean_targets: list[float] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_centers.append(float((lo + hi) / 2))
        mean_preds.append(float(probs[mask].mean()))
        mean_targets.append(float(targets[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if bin_centers:
        ax.plot(mean_preds, mean_targets, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Mean Stockfish win probability")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
