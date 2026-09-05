#!/usr/bin/env python3
"""Plot the audited two- and three-dimensional operation counts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TITLES = {
    "S2d": r"2D: $K_0(k|x-y|)$",
    "D2d": r"2D: $\partial_{\nu_y}K_0(k|x-y|)$",
    "S3d": r"3D: $e^{-k|x-y|}/|x-y|$",
    "D3d": r"3D: $\partial_{\nu_y}[e^{-k|x-y|}/|x-y|]$",
}


def _load_pickle(path: Path, *, expected_keys: set[str]) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, Mapping):
        raise TypeError(f"{path}: expected a mapping, got {type(data).__name__}")
    actual_keys = set(data)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{path}: expected kernels {sorted(expected_keys)}, "
            f"found {sorted(actual_keys)}"
        )
    return data


def load_data(path_2d: Path, path_3d: Path) -> dict[str, Any]:
    data_2d = _load_pickle(path_2d, expected_keys={"S2d", "D2d"})
    data_3d = _load_pickle(path_3d, expected_keys={"S3d", "D3d"})
    return {**data_2d, **data_3d}


def _series(kernel_data: Mapping[str, Any], name: str) -> np.ndarray:
    values = np.asarray(kernel_data[name], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional series")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return values


def make_figure(data: Mapping[str, Any]) -> plt.Figure:
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath,amssymb}")

    figure, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for axis, name in zip(axes, TITLES, strict=True):
        kernel_data = data[name]
        orders = _series(kernel_data, "orders")
        if np.any(orders != np.floor(orders)) or np.any(np.diff(orders) <= 0):
            raise ValueError(f"{name}: expansion orders must be increasing integers")

        qbx = _series(kernel_data, "qbx_after_cse")
        qbmax = _series(kernel_data, "qbmax_after_cse")
        if qbx.shape != orders.shape or qbmax.shape != orders.shape:
            raise ValueError(f"{name}: operation-count and order grids differ")

        positive_order = orders > 0
        if np.any(qbx[positive_order] <= 0) or np.any(qbmax[positive_order] <= 0):
            raise ValueError(f"{name}: plotted operation counts must be positive")
        axis.semilogy(
            orders[positive_order],
            qbx[positive_order],
            "->",
            color="#002147",
            label="QBX",
        )
        axis.semilogy(
            orders[positive_order],
            qbmax[positive_order],
            "o-",
            color="#B22222",
            label="QBMAX",
        )

        fit = orders >= 5
        if np.count_nonzero(fit) < 2 or np.any(qbmax[fit] <= 0):
            raise ValueError(f"{name}: insufficient positive data for the p>=5 fit")
        power, log_constant = np.polyfit(
            np.log(orders[fit]),
            np.log(qbmax[fit]),
            1,
        )
        axis.semilogy(
            orders[positive_order],
            np.exp(log_constant) * orders[positive_order] ** power,
            "--",
            color="#B22222",
            label=fr"Asymptotic: $p^{{{power:.1f}}}$",
        )
        axis.set_title(TITLES[name])
        axis.set_xlabel(r"expansion order $p$")
        axis.set_xticks(orders[positive_order])
        axis.grid(True, linestyle=":")
        axis.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("weighted operation count")
    figure.tight_layout()
    return figure


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-2d",
        type=Path,
        default=script_directory / "flop_count_2d.pkl",
    )
    parser.add_argument(
        "--data-3d",
        type=Path,
        default=script_directory / "flop_count_3d.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_directory / "weighted_flop.pgf",
    )
    args = parser.parse_args()

    data = load_data(args.data_2d, args.data_3d)
    figure = make_figure(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
