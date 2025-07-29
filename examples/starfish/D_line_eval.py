"""
Comparison of QBMAX vs QBX methods for close evaluation of double-layer potentials
on the starfish geometry.

This script examines the close evaluation of the double-layer potential D_k σ with
σ(y) = cos(5θ)sin(2θ) where θ := atan2(y_2, y_1) for y = (y_1, y_2) ∈ ℝ².
The analytical reference solution is obtained by using high-order QBMAX methods
on a refined mesh.

Parameters tested:
- k: [10, 20, 40, 80]
- τ: [0, 1/8, 1/4, 1/2, 3/4, 7/8, 1]
"""

import numpy as np


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
    from meshmode.mesh.generation import make_curve_mesh, starfish
    from pytential import GeometryCollection, bind, sym
    from pytential.qbx import QBXLayerPotentialSource

    mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelement + 1), target_order)
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
    normals = bind(places, sym.normal(qbx.ambient_dim, dofdesc=dofdesc))(actx).as_vector(object)
    normals_h = actx.to_numpy(flatten(normals, actx)).reshape(ambient_dim, -1)

    target_dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_STAGE1)
    normals_target = bind(
        places, sym.normal(qbx.ambient_dim, dofdesc=target_dofdesc)
    )(actx).as_vector(object)
    normals_target_h = actx.to_numpy(flatten(normals_target, actx)).reshape(
        ambient_dim, -1
    )

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

    return (sources_h, targets_h, centers_in_h, centers_out_h, weights_nodes_h, expansion_radii_h, 
            normals_h, normals_target_h, hmax_h)


def preprocess_arrays(actx, sources, targets, centers_in, centers_out, weights_nodes,
                     expansion_radii, normals, normals_target):
    """Convert numpy arrays to device arrays."""
    return (actx.from_numpy(sources), actx.from_numpy(targets), actx.from_numpy(centers_in),
            actx.from_numpy(centers_out), actx.from_numpy(weights_nodes),
            actx.from_numpy(expansion_radii), actx.from_numpy(normals), actx.from_numpy(normals_target))


def reference_solution(lam, tau, targets_h, side, expansion_radii_h, normals_target_h):
    """Generate high-accuracy reference solution using QBMAX with finely refined mesh."""
    ref_element = 200
    target_order_ref = 7
    expn_order_ref = 7
    upsampling_factor_ref = 5

    geometry_data_ref = setup_geometry(ref_element, expn_order_ref, target_order_ref, 
                                      upsampling_factor_ref)
    sources_h_ref = geometry_data_ref[0]
    weights_nodes_h_ref = geometry_data_ref[4]
    
    angle = np.arctan2(sources_h_ref[1], sources_h_ref[0])
    sigma_ref = np.cos(5 * angle) * np.sin(2 * angle) * weights_nodes_h_ref
    strengths_ref = (actx.from_numpy(sigma_ref),)

    centers_ref = actx.from_numpy(targets_h + side / 2 * expansion_radii_h * normals_target_h)
    normals_ref = geometry_data_ref[-3]
    
    from sumpy.kernel import YukawaKernel, DirectionalSourceDerivative

    knl = YukawaKernel(2)
    knl = DirectionalSourceDerivative(knl, dir_vec_name="dsource_vec")
    asym_knl = asym_yukawa(2)
    extra_kwargs = {"lam": lam, 'dsource_vec': normals_ref}
    from sumpy.expansion.local import AsymptoticDividingLineTaylorExpansion

    asymexpn_ref = AsymptoticDividingLineTaylorExpansion(knl, asym_knl, expn_order_ref, tau=tau)

    from sumpy.qbx import LayerPotential

    lplot_asym = LayerPotential(
        actx.context, expansion=asymexpn_ref, source_kernels=(knl,), target_kernels=(knl,)
    )

    _, (result_ref,) = lplot_asym(
        actx.queue,
        targets=actx.from_numpy(targets_h),
        sources=actx.from_numpy(sources_h_ref),
        strengths=strengths_ref,
        centers=centers_ref,
        expansion_radii=actx.from_numpy(expansion_radii_h),
        **extra_kwargs,
    )
    result_ref = actx.to_numpy(result_ref) * np.exp(-lam * expansion_radii_h * (1 - tau))

    return result_ref


def evaluate_expansion(expansion_type, knl, asym_knl, expn_order, tau, lam,
                      targets, sources, expansion_radii, centers_in, centers_out,
                      weights_nodes_h, sources_h, normals):
    """Evaluate QBMAX and QBX expansion."""
    from sumpy.expansion.local import (
        AsymptoticDividingLineTaylorExpansion,
        LineTaylorLocalExpansion,
    )
    from sumpy.qbx import LayerPotentialMatrixGenerator

    extra_kwargs = {"lam": lam, 'dsource_vec': normals}

    if expansion_type == "QBMAX":
        expansion = AsymptoticDividingLineTaylorExpansion(knl, asym_knl, expn_order, tau=tau)
    elif expansion_type == "QBX":
        expansion = LineTaylorLocalExpansion(knl, expn_order, tau=tau)
    else:
        msg = f"Unknown expansion type: {expansion_type}"
        raise ValueError(msg)

    mat_gen = LayerPotentialMatrixGenerator(
        actx.context, expansion=expansion, source_kernels=(knl,), target_kernels=(knl,)
    )

    angle = np.arctan2(sources_h[1, :], sources_h[0, :])
    sigma = np.cos(5 * angle) * np.sin(2 * angle) * weights_nodes_h

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


def run_comparison(lams, taus, nelement=40, target_order=5, expn_order=5, upsampling_factor=4):
    """Run the QBMAX vs QBX comparison."""
    geometry_data = setup_geometry(nelement, expn_order, target_order, upsampling_factor)
    (sources_h, targets_h, centers_in_h, centers_out_h,
     weights_nodes_h, expansion_radii_h, normals, normals_target_h, hmax_h) = geometry_data

    processed_arrays = preprocess_arrays(
        actx, sources_h, targets_h, centers_in_h, centers_out_h,
        weights_nodes_h, expansion_radii_h, normals, normals_target_h
    )
    (sources, targets, centers_in, centers_out, weights_nodes,
     expansion_radii, normal, normal_target) = processed_arrays

    from sumpy.kernel import YukawaKernel, DirectionalSourceDerivative

    knl = YukawaKernel(2)
    knl = DirectionalSourceDerivative(knl, dir_vec_name="dsource_vec")
    asym_knl = asym_yukawa(2)

    qbmax_results = {}
    qbx_results = {}

    for lam in lams:
        qbmax_results[lam] = {}
        qbx_results[lam] = {}

        for tau in taus:
            utrue_vec_in = reference_solution(lam, tau, targets_h, -2, expansion_radii_h, 
                                            normals_target_h)
            utrue_vec_out = reference_solution(lam, tau, targets_h, 2, expansion_radii_h, 
                                             normals_target_h)

            # QBMAX evaluation
            qbmax_eval = evaluate_expansion("QBMAX", knl, asym_knl, expn_order, tau, lam,
                                           targets, sources, expansion_radii, centers_in, centers_out,
                                           weights_nodes_h, sources_h, normals)

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
            qbx_eval = evaluate_expansion("QBX", knl, asym_knl, expn_order, tau, lam,
                                         targets, sources, expansion_radii, centers_in, centers_out,
                                         weights_nodes_h, sources_h, normals)

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
            "mathtext.fontset": "cm",
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

    plot_taus = taus[1:]

    for i, lam in enumerate(lams):
        qbx_in = [qbx_results[lam][tau]["abs_err_in"] for tau in plot_taus]
        qbx_out = [qbx_results[lam][tau]["abs_err_out"] for tau in plot_taus]
        qbmax_in = [qbmax_results[lam][tau]["abs_err_in"] for tau in plot_taus]
        qbmax_out = [qbmax_results[lam][tau]["abs_err_out"] for tau in plot_taus]

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 5]

        axs[0].plot(np.log10(plot_taus), np.log10(qbx_in), marker=qbx_marker, linestyle="--",
                    color=color, label=f"QBX, $k={lam}$", markersize=6)
        axs[0].plot(np.log10(plot_taus), np.log10(qbmax_in), marker=qbmax_marker, linestyle="-",
                    color=color, label=f"QBMAX, $k={lam}$", markersize=6)
        axs[1].plot(np.log10(plot_taus), np.log10(qbx_out), marker=qbx_marker, linestyle="--",
                    color=color, label=f"QBX, $k={lam}$", markersize=6)
        axs[1].plot(np.log10(plot_taus), np.log10(qbmax_out), marker=qbmax_marker, linestyle="-",
                    color=color, label=f"QBMAX, $k={lam}$", markersize=6)

    axs[0].set_ylabel(r"$\log_{10}(\mathrm{err}^{-}(\tau))$")
    axs[1].set_ylabel(r"$\log_{10}(\mathrm{err}^{+}(\tau))$")

    for ax in axs:
        ax.set_xlabel(r"$\log_{10}(\tau)$", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params()

    plt.suptitle(rf"Close Evaluation of $\mathcal{{D}}_{{k}}\sigma$: $N={nelement}$, "
                 rf"$p={expn_order}$, $q={target_order + 1}$, $\kappa={upsampling_factor}$",
                 fontsize=14, y=0.95)

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0),
               ncol=4, frameon=True, handlelength=2.5)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    return fig


import pyopencl as cl
from meshmode.array_context import PyOpenCLArrayContext

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)
actx = PyOpenCLArrayContext(queue)

lams = [10, 20, 40, 80]
taus = [0, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 1]
nelement = 120
expn_order = 6
target_order = 6
upsampling_factor = 5

print("Starting QBMAX vs QBX comparison...")
print(f"Parameters: k = {lams}, τ = {taus}")

qbmax_results, qbx_results = run_comparison(
    lams, taus, nelement, target_order, expn_order, upsampling_factor
)

print("Comparison completed. Visualizing results...")
fig = visualize_results(qbmax_results, qbx_results, lams, taus,
                       nelement, expn_order, target_order, upsampling_factor)

import matplotlib.pyplot as plt
plt.show()