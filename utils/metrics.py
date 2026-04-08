from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

ArrayLike = np.ndarray | torch.Tensor

EVAL_METRIC_DICT = {
    "interpretability": "Interpretability",
    "modularity_score": "Modularity",
    "mig": "MIG",
    "SAP_score": "SAP",
    "Corr_score": "SCC",
}


@dataclass(frozen=True)
class DisentanglementInputs:
    z: np.ndarray
    attributes: np.ndarray


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _prepare_inputs(z: ArrayLike, attributes: ArrayLike) -> DisentanglementInputs:
    z_np = _to_numpy(z).astype(np.float64)
    a_np = _to_numpy(attributes).astype(np.float64)

    if z_np.ndim != 2:
        raise ValueError(f"`z` must be 2D, got shape={z_np.shape}")
    if a_np.ndim != 2:
        raise ValueError(f"`attributes` must be 2D, got shape={a_np.shape}")
    if z_np.shape[0] != a_np.shape[0]:
        raise ValueError(
            "z and attributes must have the same number of samples: "
            f"{z_np.shape[0]} != {a_np.shape[0]}"
        )

    return DisentanglementInputs(z=z_np, attributes=a_np)


def normalize_data(data: np.ndarray, mean: np.ndarray | None = None, stddev: np.ndarray | None = None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if stddev is None:
        stddev = np.std(data, axis=0)
    return (data - mean[np.newaxis, :]) / stddev[np.newaxis, :], mean, stddev


def collect_latent_metric_data(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device | str | None = None,
    use_mu: bool = True,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect (z, attributes) for disentanglement metrics.

    Expected batch keys:
    - "input_ids": token ids passed to encoder
    - "attributes": ground-truth factors, shape (B, L)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Cannot infer device from model with no parameters.") from exc
    device = torch.device(device)

    z_chunks: list[np.ndarray] = []
    attr_chunks: list[np.ndarray] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if "input_ids" not in batch:
                    raise KeyError("Batch is missing required key 'input_ids'.")
                if "attributes" not in batch:
                    raise KeyError("Batch is missing required key 'attributes'.")

                src = batch["input_ids"].to(device)
                attrs = batch["attributes"].to(device).float()

                mu, logvar = model.encoder(src)
                z = mu if use_mu else model.reparametrize(mu, logvar)

                z_chunks.append(z.detach().cpu().numpy().astype(np.float64))
                attr_chunks.append(attrs.detach().cpu().numpy().astype(np.float64))
    finally:
        if was_training:
            model.train()

    if len(z_chunks) == 0:
        raise ValueError("No samples collected from dataloader.")

    return np.concatenate(z_chunks, axis=0), np.concatenate(attr_chunks, axis=0)


def discrete_mutual_info(latent_codes: np.ndarray, attributes: np.ndarray) -> np.ndarray:
    num_codes = latent_codes.shape[1]
    num_attributes = attributes.shape[1]
    m = np.zeros([num_codes, num_attributes], dtype=np.float64)
    for i in range(num_codes):
        for j in range(num_attributes):
            m[i, j] = mutual_info_score(attributes[:, j], latent_codes[:, i])
    return m


def continuous_mutual_info(latent_codes: np.ndarray, attributes: np.ndarray, show_progress: bool = False) -> np.ndarray:
    num_codes = latent_codes.shape[1]
    num_attributes = attributes.shape[1]
    m = np.zeros([num_codes, num_attributes], dtype=np.float64)
    for i in tqdm(range(num_attributes), desc="MI(attr|z)", leave=False, disable=not show_progress):
        m[:, i] = mutual_info_regression(latent_codes, attributes[:, i], random_state=0)
    return m


def discrete_entropy(attributes: np.ndarray) -> np.ndarray:
    num_factors = attributes.shape[1]
    h = np.zeros(num_factors, dtype=np.float64)
    for j in range(num_factors):
        h[j] = mutual_info_score(attributes[:, j], attributes[:, j])
    return h


def continuous_entropy(attributes: np.ndarray, show_progress: bool = False) -> np.ndarray:
    num_factors = attributes.shape[1]
    h = np.zeros(num_factors, dtype=np.float64)
    for j in tqdm(range(num_factors), desc="H(attr)", leave=False, disable=not show_progress):
        h[j] = float(
            mutual_info_regression(attributes[:, j].reshape(-1, 1), attributes[:, j], random_state=0)[0]
        )
    return h


def compute_interpretability_metric(
    latent_codes: np.ndarray,
    attributes: np.ndarray,
    attr_list: Sequence[str],
    show_progress: bool = False,
):
    interpretability_metrics: dict[str, tuple[int, float]] = {}
    total = 0.0
    for i, attr_name in tqdm(
        list(enumerate(attr_list)),
        desc="Interpretability",
        leave=False,
        disable=not show_progress,
    ):
        attr_labels = attributes[:, i]
        mutual_info = mutual_info_regression(latent_codes, attr_labels, random_state=0)
        dim = int(np.argmax(mutual_info))

        reg = LinearRegression().fit(latent_codes[:, dim : dim + 1], attr_labels)
        score = float(reg.score(latent_codes[:, dim : dim + 1], attr_labels))
        interpretability_metrics[str(attr_name)] = (dim, score)
        total += score

    interpretability_metrics["mean"] = (-1, total / max(1, len(attr_list)))
    return interpretability_metrics


def compute_mig(latent_codes: np.ndarray, attributes: np.ndarray, show_progress: bool = False) -> dict[str, float]:
    m = continuous_mutual_info(latent_codes, attributes, show_progress=show_progress)
    entropy = continuous_entropy(attributes, show_progress=show_progress)
    sorted_m = np.sort(m, axis=0)[::-1]
    if sorted_m.shape[0] < 2:
        return {"mig": 0.0}
    denom = np.where(entropy > 1e-12, entropy, np.inf)
    return {"mig": float(np.mean((sorted_m[0, :] - sorted_m[1, :]) / denom))}


def _modularity(mutual_information: np.ndarray) -> float:
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.0)

    modularity = np.zeros_like(max_squared_mi, dtype=np.float64)
    nonzero = max_squared_mi > 0.0
    modularity[nonzero] = 1.0 - (numerator[nonzero] / denominator[nonzero])
    return float(np.mean(modularity))


def compute_modularity(latent_codes: np.ndarray, attributes: np.ndarray, show_progress: bool = False) -> dict[str, float]:
    mi = continuous_mutual_info(latent_codes, attributes, show_progress=show_progress)
    return {"modularity_score": _modularity(mi)}


def _compute_correlation_matrix(latent_codes: np.ndarray, attributes: np.ndarray, show_progress: bool = False) -> np.ndarray:
    num_latent_codes = latent_codes.shape[1]
    num_attributes = attributes.shape[1]
    score_matrix = np.zeros([num_latent_codes, num_attributes], dtype=np.float64)
    for i in tqdm(range(num_latent_codes), desc="Spearman", leave=False, disable=not show_progress):
        for j in range(num_attributes):
            rho, p = spearmanr(latent_codes[:, i], attributes[:, j])
            if rho is None or np.isnan(rho) or p is None or np.isnan(p):
                score_matrix[i, j] = 0.0
            elif p <= 0.05:
                score_matrix[i, j] = float(abs(rho))
            else:
                score_matrix[i, j] = 0.0
    return score_matrix


def compute_correlation_score(latent_codes: np.ndarray, attributes: np.ndarray, show_progress: bool = False) -> dict[str, float]:
    corr_matrix = _compute_correlation_matrix(latent_codes, attributes, show_progress=show_progress)
    return {"Corr_score": float(np.mean(np.max(corr_matrix, axis=0)))}


def _compute_score_matrix(latent_codes: np.ndarray, attributes: np.ndarray, show_progress: bool = False) -> np.ndarray:
    num_latent_codes = latent_codes.shape[1]
    num_attributes = attributes.shape[1]
    score_matrix = np.zeros([num_latent_codes, num_attributes], dtype=np.float64)
    for i in tqdm(range(num_latent_codes), desc="SAP", leave=False, disable=not show_progress):
        for j in range(num_attributes):
            cov = np.cov(latent_codes[:, i], attributes[:, j], ddof=1)
            cov_sq = cov[0, 1] ** 2
            var_mu = cov[0, 0]
            var_y = cov[1, 1]
            if var_mu > 1e-12 and var_y > 1e-12:
                score_matrix[i, j] = float(cov_sq / (var_mu * var_y))
            else:
                score_matrix[i, j] = 0.0
    return score_matrix


def _compute_avg_diff_top_two(matrix: np.ndarray) -> float:
    sorted_matrix = np.sort(matrix, axis=0)
    if sorted_matrix.shape[0] < 2:
        return 0.0
    return float(np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :]))


def compute_sap_score(latent_codes: np.ndarray, attributes: np.ndarray, show_progress: bool = False) -> dict[str, float]:
    score_matrix = _compute_score_matrix(latent_codes, attributes, show_progress=show_progress)
    return {"SAP_score": _compute_avg_diff_top_two(score_matrix)}


def compute_disentanglement_metrics(
    latent_codes: ArrayLike,
    attributes: ArrayLike,
    attr_names: Sequence[str] | None = None,
    normalize: bool = False,
    show_progress: bool = False,
    metrics: Sequence[str] = ("interpretability", "mig", "modularity_score", "SAP_score", "Corr_score"),
) -> dict[str, float]:
    """
    General entry point: compute a chosen set of metrics by name.

    Metric names:
    - "interpretability"
    - "mig"
    - "modularity_score"
    - "SAP_score"
    - "Corr_score"
    """
    inp = _prepare_inputs(latent_codes, attributes)

    z = inp.z
    y = inp.attributes
    if normalize:
        z, _, _ = normalize_data(z)
        y, _, _ = normalize_data(y)

    if attr_names is None:
        attr_names = [f"attr_{i}" for i in range(y.shape[1])]
    if len(attr_names) != y.shape[1]:
        raise ValueError(f"len(attr_names) must match attributes.shape[1]={y.shape[1]}")

    out: dict[str, float] = {}
    for name in metrics:
        if name == "interpretability":
            details = compute_interpretability_metric(z, y, attr_names, show_progress=show_progress)
            out["interpretability"] = float(details["mean"][1])
        elif name == "mig":
            out["mig"] = float(compute_mig(z, y, show_progress=show_progress)["mig"])
        elif name == "modularity_score":
            out["modularity_score"] = float(compute_modularity(z, y, show_progress=show_progress)["modularity_score"])
        elif name == "SAP_score":
            out["SAP_score"] = float(compute_sap_score(z, y, show_progress=show_progress)["SAP_score"])
        elif name == "Corr_score":
            out["Corr_score"] = float(compute_correlation_score(z, y, show_progress=show_progress)["Corr_score"])
        else:
            raise ValueError(f"Unknown metric name: {name}")
    return out


def interpretability_score(z: ArrayLike, attributes: ArrayLike, attr_names: Sequence[str] | None = None) -> float:
    return float(compute_disentanglement_metrics(z, attributes, attr_names=attr_names, metrics=("interpretability",))["interpretability"])


def mig_score(z: ArrayLike, attributes: ArrayLike) -> float:
    return float(compute_disentanglement_metrics(z, attributes, metrics=("mig",))["mig"])


def modularity_score(z: ArrayLike, attributes: ArrayLike) -> float:
    return float(compute_disentanglement_metrics(z, attributes, metrics=("modularity_score",))["modularity_score"])


def sap_score(z: ArrayLike, attributes: ArrayLike) -> float:
    return float(compute_disentanglement_metrics(z, attributes, metrics=("SAP_score",))["SAP_score"])


def spearman_correlation_score(z: ArrayLike, attributes: ArrayLike) -> float:
    return float(compute_disentanglement_metrics(z, attributes, metrics=("Corr_score",))["Corr_score"])


def all_disentanglement_metrics(z: ArrayLike, attributes: ArrayLike, attr_names: Sequence[str] | None = None) -> dict[str, float]:
    """
    Compatibility wrapper returning the project keys:
    - interpretability, mig, modularity, sap, spearman
    """
    raw = compute_disentanglement_metrics(
        latent_codes=z,
        attributes=attributes,
        attr_names=attr_names,
        metrics=("interpretability", "mig", "modularity_score", "SAP_score", "Corr_score"),
    )
    return {
        "interpretability": raw["interpretability"],
        "mig": raw["mig"],
        "modularity": raw["modularity_score"],
        "sap": raw["SAP_score"],
        "spearman": raw["Corr_score"],
    }


def all_disentanglement_metrics_from_model(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device | str | None = None,
    use_mu: bool = True,
    max_batches: int | None = None,
    attr_names: Sequence[str] | None = None,
) -> dict[str, float]:
    z, attributes = collect_latent_metric_data(
        model=model,
        dataloader=dataloader,
        device=device,
        use_mu=use_mu,
        max_batches=max_batches,
    )
    return all_disentanglement_metrics(z=z, attributes=attributes, attr_names=attr_names)
