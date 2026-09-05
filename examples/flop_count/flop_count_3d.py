#!/usr/bin/env python3
"""Generate weighted operation counts for the three-dimensional kernels."""

from __future__ import annotations

from pathlib import Path
import pickle

import sympy as sp


EXPONENTIAL_WEIGHT = 10
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "flop_count_3d.pkl"


def weighted_count_ops(expression: sp.Expr) -> int:
    count = 0
    for node in sp.preorder_traversal(expression):
        if node.func == sp.exp:
            count += EXPONENTIAL_WEIGHT
        elif isinstance(node, (sp.Add, sp.Mul)):
            # SymPy flattens n-ary sums and products into a single node.
            count += len(node.args) - 1
        elif isinstance(node, (sp.Pow, sp.Function)):
            count += 1
    return count


def generate_counts(max_order: int = 8) -> dict[str, dict[str, list[int]]]:
    if max_order < 0:
        raise ValueError("maximum expansion order must be nonnegative")
    orders = list(range(max_order + 1))

    x0, x1, x2, y0, y1, y2, c0, c1, c2, ny0, ny1, ny2 = sp.symbols(
        "x0 x1 x2 y0 y1 y2 c0 c1 c2 ny0 ny1 ny2"
    )
    x = sp.Matrix([x0, x1, x2])
    y = sp.Matrix([y0, y1, y2])
    c = sp.Matrix([c0, c1, c2])
    ny = sp.Matrix([ny0, ny1, ny2])
    k, tau = sp.symbols("k tau")

    dist_xc = sp.sqrt(((x - c).T @ (x - c))[0, 0])
    line = (c - y) + tau * (x - c)
    dist_line = sp.sqrt((line.T @ line)[0, 0])

    single_layer = sp.exp(-k * dist_line) / dist_line
    double_layer = sum(single_layer.diff(y[i]) * ny[i] for i in range(3))
    kernels = {"S3d": single_layer, "D3d": double_layer}

    data: dict[str, dict[str, list[int]]] = {}
    for name, kernel in kernels.items():
        kernel_data = {
            "orders": orders,
            "qbx_before_cse": [],
            "qbx_after_cse": [],
            "qbmax_before_cse": [],
            "qbmax_after_cse": [],
        }

        for order in orders:
            qbx = sum(
                kernel.diff(tau, i).subs(tau, 0)
                / sp.factorial(i)
                * tau**i
                for i in range(order + 1)
            )
            qbmax = sp.exp(-k * (1 - tau) * dist_xc) * sum(
                (kernel * sp.exp(k * (1 - tau) * dist_xc))
                .diff(tau, i)
                .subs(tau, 0)
                / sp.factorial(i)
                * tau**i
                for i in range(order + 1)
            )

            qbx_replacements, qbx_reduced = sp.cse(qbx)
            qbmax_replacements, qbmax_reduced = sp.cse(qbmax)

            kernel_data["qbx_before_cse"].append(weighted_count_ops(qbx))
            kernel_data["qbx_after_cse"].append(
                sum(weighted_count_ops(rhs) for _, rhs in qbx_replacements)
                + sum(weighted_count_ops(item) for item in qbx_reduced)
            )
            kernel_data["qbmax_before_cse"].append(weighted_count_ops(qbmax))
            kernel_data["qbmax_after_cse"].append(
                sum(weighted_count_ops(rhs) for _, rhs in qbmax_replacements)
                + sum(weighted_count_ops(item) for item in qbmax_reduced)
            )

            print(name, order)
            print(
                "QBX:",
                kernel_data["qbx_before_cse"][-1],
                kernel_data["qbx_after_cse"][-1],
            )
            print(
                "QBMAX:",
                kernel_data["qbmax_before_cse"][-1],
                kernel_data["qbmax_after_cse"][-1],
            )

        data[name] = kernel_data

    return data


if __name__ == "__main__":
    data = generate_counts()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("wb") as stream:
        pickle.dump(data, stream, protocol=4)
    print(f"wrote {DEFAULT_OUTPUT}")
