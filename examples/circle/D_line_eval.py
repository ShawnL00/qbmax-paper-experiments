"""
Comparison of QBMAX vs QBX methods for close evaluation of double-layer potentials
on the unit circle.

This script examines the close evaluation of the double-layer potential D_k σ with
σ(y) = e^{imθ} where m=7 and θ := atan2(y_2, y_1) for y = (y_1, y_2) ∈ ℝ².

The analytical solution is given by:
    u_true(x) := D_k σ(x) = {
        ke^{imθ_x} I_m(k|x|) K'_m(k),     |x| < 1
        ke^{imθ_x} I'_m(k) K_m(k|x|),     |x| > 1
    }
where θ_x = atan2(x_2, x_1), I_m and K_m are modified Bessel functions.

Parameters tested:
- k: [10, 20, 40, 80]
- τ: [0, 1/8, 1/4, 1/2, 3/4, 7/8, 1]
"""

import numpy as np


# σ(y) = e^{imθ}
m = 7


def asym_yukawa(dim, lam=None):
    """Asymptotic Yukawa kernel function."""
    from pymbolic import primitives, var
    from sumpy.symbolic import SpatialConstant, pymbolic_real_norm_2

    b = pymbolic_real_norm_2(primitives.make_sym_vector("b", dim))

    if lam:
        expr = var("exp")(-lam * b * (1 - var("tau")))
    else:
        lam = SpatialConstant("lam")
        expr = var("exp")(-lam * b * (1 - var("tau")))

    return expr


def setup_geometry(nelement, expn_order, target_order, upsampling_factor=1):
    """Set up mesh, discretization, and QBX source."""
    from arraycontext import flatten
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )
    from meshmode.mesh.generation import circle, make_curve_mesh
    from pytential import GeometryCollection, bind, sym
    from pytential.qbx import QBXLayerPotentialSource

    mesh = make_curve_mesh(circle, np.linspace(0, 1, nelement + 1), target_order)

    pre_density_discr = Discretization(
        actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order)
    )

    qbx = QBXLayerPotentialSource(
        pre_density_discr, upsampling_factor * target_order, expn_order, fmm_order=False
    )

    places = GeometryCollection({"qbx": qbx}, auto_where=("qbx"))

    target_discr = places.get_discretization("qbx", sym.QBX_SOURCE_STAGE1)
    source_discr = places.get_discretization("qbx", sym.QBX_SOURCE_QUAD_STAGE2)

    targets = target_discr.nodes()
    sources = source_discr.nodes()

    ambient_dim = 2
    dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_QUAD_STAGE2)

    normals = bind(
        places, sym.normal(qbx.ambient_dim, dofdesc=dofdesc)
    )(actx).as_vector(object)
    normals_h = actx.to_numpy(flatten(normals, actx)).reshape(ambient_dim, -1)

    expansion_radii = bind(places, sym.expansion_radii(ambient_dim))(actx)
    centers_in = bind(places, sym.expansion_centers(qbx.ambient_dim, -1))(actx)
    centers_out = bind(places, sym.expansion_centers(qbx.ambient_dim, +1))(actx)

    weights_nodes = bind(
        places, sym.weights_and_area_elements(ambient_dim=2, dim=1, dofdesc=dofdesc)
    )(actx)
    hmax = bind(places, sym.h_max(qbx.ambient_dim))(actx)

    targets_h = actx.to_numpy(flatten(targets, actx)).reshape(ambient_dim, -1)
    sources_h = actx.to_numpy(flatten(sources, actx)).reshape(ambient_dim, -1)

    expansion_radii_h = actx.to_numpy(flatten(expansion_radii, actx))
    centers_in_h = actx.to_numpy(flatten(centers_in, actx)).reshape(ambient_dim, -1)
    centers_out_h = actx.to_numpy(flatten(centers_out, actx)).reshape(ambient_dim, -1)
    weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))
    hmax_h = actx.to_numpy(flatten(hmax, actx))

    return (sources_h, targets_h, centers_in_h, centers_out_h,
            weights_nodes_h, expansion_radii_h, normals_h, hmax_h)


def preprocess_arrays(actx, sources, targets, centers_in, centers_out, weights_nodes,
                     expansion_radii, normal):
    """Convert numpy arrays to device arrays."""
    return (actx.from_numpy(sources), actx.from_numpy(targets),
            actx.from_numpy(centers_in), actx.from_numpy(centers_out),
            actx.from_numpy(weights_nodes), actx.from_numpy(expansion_radii),
            actx.from_numpy(normal))


def analytical_solution(lam, r, tau, targets_h, m, side):
    """Compute analytical solution for validation."""
    import mpmath

    mpmath.mp.dps = 25

    angles = np.arctan2(targets_h[1, :], targets_h[0, :])
    n_points = len(angles)
    result = np.zeros(n_points, dtype=np.complex128)

    for i in range(n_points):
        r_i = float(r[i])

        if side == -1:
            coeff = (lam * mpmath.besseli(m, lam * (1 - (1 - tau) * r_i)) * -1
                     * (mpmath.besselk(m - 1, lam) + mpmath.besselk(m + 1, lam)) / 2)
        else:
            coeff = (lam * mpmath.besselk(m, lam * (1 + (1 - tau) * r_i))
                     * (mpmath.besseli(m - 1, lam) + mpmath.besseli(m + 1, lam)) / 2)

        result[i] = coeff * np.exp(1j * m * angles[i])

    return result


def evaluate_expansion(expansion_type, knl, asym_knl, expn_order, tau, lam,
                      targets, sources, expansion_radii, centers_in, centers_out,
                      weights_nodes_h, sources_h, normal):
    """Evaluate QBMAX and QBX Expansion."""
    from sumpy.expansion.local import (
        AsymptoticDividingLineTaylorExpansion,
        LineTaylorLocalExpansion,
    )
    from sumpy.qbx import LayerPotentialMatrixGenerator

    extra_kwargs = {"lam": lam, "dsource_vec": normal}

    if expansion_type == "QBMAX":
        expansion = AsymptoticDividingLineTaylorExpansion(
            knl, asym_knl, expn_order, tau=tau
        )
    elif expansion_type == "QBX":
        expansion = LineTaylorLocalExpansion(knl, expn_order, tau=tau)
    else:
        msg = f"Unknown expansion type: {expansion_type}"
        raise ValueError(msg)

    mat_gen = LayerPotentialMatrixGenerator(
        actx.context, expansion=expansion, source_kernels=(knl,), target_kernels=(knl,)
    )

    angle = np.arctan2(sources_h[1, :], sources_h[0, :])
    sigma = np.exp(1j * m * angle) * weights_nodes_h

    results = {}

    for side, centers in [("inner", centers_in), ("outer", centers_out)]:
        _, (mat,) = mat_gen(
            actx.queue,
            targets=targets,
            sources=sources,
            expansion_radii=expansion_radii,
            centers=centers,
            **extra_kwargs,
        )

        mat = actx.to_numpy(mat)
        weighted_mat = mat * sigma[None, :]
        mat_eval = np.sum(weighted_mat, axis=1)

        if expansion_type == "QBMAX":
            expansion_radii_h = actx.to_numpy(expansion_radii)
            mat_eval *= np.exp(-lam * expansion_radii_h * (1 - tau))

        results[side] = mat_eval

    return results


def run_comparison(lams, taus, nelement, target_order, expn_order, upsampling_factor):
    """Run the QBMAX vs QBX comparison."""
    geometry_data = setup_geometry(nelement, expn_order, target_order,
                                   upsampling_factor)
    (sources_h, targets_h, centers_in_h, centers_out_h,
     weights_nodes_h, expansion_radii_h, normals_h, _hmax_h) = geometry_data

    processed_arrays = preprocess_arrays(
        actx, sources_h, targets_h, centers_in_h, centers_out_h,
        weights_nodes_h, expansion_radii_h, normals_h
    )
    (sources, targets, centers_in, centers_out, _weights_nodes,
     expansion_radii, normal) = processed_arrays

    from sumpy.kernel import DirectionalSourceDerivative, YukawaKernel

    knl = YukawaKernel(2)
    knl = DirectionalSourceDerivative(knl, dir_vec_name="dsource_vec")
    asym_knl = asym_yukawa(2)

    qbmax_results = {}
    qbx_results = {}

    for lam in lams:
        qbmax_results[lam] = {}
        qbx_results[lam] = {}

        for tau in taus:
            utrue_vec_in = analytical_solution(lam, expansion_radii_h, tau,
                                               targets_h, m, -1)
            utrue_vec_out = analytical_solution(lam, expansion_radii_h, tau,
                                                targets_h, m, 1)

            # QBMAX evaluation
            qbmax_eval = evaluate_expansion("QBMAX", knl, asym_knl, expn_order, tau,
                                            lam, targets, sources, expansion_radii,
                                            centers_in, centers_out, weights_nodes_h,
                                            sources_h, normal)

            err_in = np.max(np.abs(qbmax_eval["inner"] - utrue_vec_in))
            err_out = np.max(np.abs(qbmax_eval["outer"] - utrue_vec_out))

            qbmax_results[lam][tau] = {
                "inner": qbmax_eval["inner"],
                "outer": qbmax_eval["outer"],
                "utrue_in": utrue_vec_in,
                "utrue_out": utrue_vec_out,
                "abs_err_in": err_in,
                "abs_err_out": err_out,
            }

            # QBX evaluation
            qbx_eval = evaluate_expansion("QBX", knl, asym_knl, expn_order, tau,
                                          lam, targets, sources, expansion_radii,
                                          centers_in, centers_out, weights_nodes_h,
                                          sources_h, normal)

            err_in = np.max(np.abs(qbx_eval["inner"] - utrue_vec_in))
            err_out = np.max(np.abs(qbx_eval["outer"] - utrue_vec_out))

            qbx_results[lam][tau] = {
                "inner": qbx_eval["inner"],
                "outer": qbx_eval["outer"],
                "utrue_in": utrue_vec_in,
                "utrue_out": utrue_vec_out,
                "abs_err_in": err_in,
                "abs_err_out": err_out,
            }

    print("Comparison completed")
    return qbmax_results, qbx_results


def visualize_results(qbmax_results, qbx_results, lams, taus, nelement, expn_order,
                     target_order, upsampling_factor):
    """Create comparison plot."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath,amsfonts,lmodern}",
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
            "axes.prop_cycle": cycler(
                color=["#002147", "#B22222", "#708090", "#228B22", "#6A0DAD"]
            ),
        }
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    qbx_marker = ">"
    qbmax_marker = "o"

    plot_taus = taus[2:]

    for i, lam in enumerate(lams):
        qbx_in = [qbx_results[lam][tau]["abs_err_in"] for tau in plot_taus]
        qbx_out = [qbx_results[lam][tau]["abs_err_out"] for tau in plot_taus]
        qbmax_in = [qbmax_results[lam][tau]["abs_err_in"] for tau in plot_taus]
        qbmax_out = [qbmax_results[lam][tau]["abs_err_out"] for tau in plot_taus]

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 5]

        axs[0].plot(np.log10(plot_taus), np.log10(qbx_in), marker=qbx_marker,
                    linestyle="--", color=color, label=f"QBX, $k={lam}$", markersize=6)
        axs[0].plot(np.log10(plot_taus), np.log10(qbmax_in), marker=qbmax_marker,
                    linestyle="-", color=color, label=f"QBMAX, $k={lam}$", markersize=6)
        axs[1].plot(np.log10(plot_taus), np.log10(qbx_out), marker=qbx_marker,
                    linestyle="--", color=color, label=f"QBX, $k={lam}$", markersize=6)
        axs[1].plot(np.log10(plot_taus), np.log10(qbmax_out), marker=qbmax_marker,
                    linestyle="-", color=color, label=f"QBMAX, $k={lam}$", markersize=6)

    axs[0].set_ylabel(r"$\log_{10}(\mathrm{err}^{-}(\tau))$")
    axs[1].set_ylabel(r"$\log_{10}(\mathrm{err}^{+}(\tau))$")

    for ax in axs:
        ax.set_xlabel(r"$\log_{10}(\tau)$", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params()

    plt.suptitle(rf"Close Evaluation of $\mathcal{{D}}_{{k}}\sigma$: $N={nelement}$, "
                 rf"$p={expn_order}$, $q={target_order + 1}$, "
                 rf"$\kappa={upsampling_factor}$")

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0),
               ncol=4, frameon=True, handlelength=2.5)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


def main():
    # Parameters
    lams = [10, 20, 40, 80]
    taus = [0, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 1]
    nelement = 80
    expn_order = 4
    target_order = 6
    upsampling_factor = 4

    print("Starting QBMAX vs QBX comparison...")
    print(f"Parameters: k = {lams}, τ = {taus}")

    qbmax_results, qbx_results = run_comparison(
        lams, taus, nelement, target_order, expn_order, upsampling_factor
    )

    print("Visualizing results...")
    visualize_results(
        qbmax_results, qbx_results,
        lams, taus, nelement, expn_order, target_order, upsampling_factor
    )


if __name__ == "__main__":

    import pyopencl as cl
    from meshmode.array_context import PyOpenCLArrayContext

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)
    main()
