"""
Jump relations verification for the double layer potential Dₖσ
on the starfish domain, where the density function is defined as:
σ(x) := sin(4πx₁)cos(2πx₂)cos(5x₂) + 2 for x = (x₁, x₂).
"""

import numpy as np


def sigma_fun(x):
    """σ(x) = sin(4πx₁)cos(2πx₂)cos(5x₂) + 2"""
    return np.sin(4*np.pi*x[0]) * np.cos(2*np.pi*x[1]) * np.cos(5*x[1]) + 2


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
    normals = bind(places, sym.normal(qbx.ambient_dim, dofdesc=dofdesc))(actx)
    normals = normals.as_vector(object)
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
    return (sources_h, targets_h, centers_in_h, centers_out_h, weights_nodes_h,
            expansion_radii_h, normals_h, normals_target_h, hmax_h)


def preprocess_arrays(sources, targets, centers_in, centers_out, weights_nodes,
                     expansion_radii, normals, normals_target):
    """Convert numpy arrays to device arrays."""
    return (actx.from_numpy(sources), actx.from_numpy(targets),
            actx.from_numpy(centers_in), actx.from_numpy(centers_out),
            actx.from_numpy(weights_nodes), actx.from_numpy(expansion_radii),
            actx.from_numpy(normals), actx.from_numpy(normals_target))


def main():
    from sumpy.kernel import DirectionalSourceDerivative, YukawaKernel
    base_knl = YukawaKernel(2)
    knl = DirectionalSourceDerivative(base_knl, dir_vec_name="dsource_vec")
    asym_knl = asym_yukawa(2)

    # Parameters
    expn_order = 6
    target_order = 6
    upsampling_factor = 4
    nelements = [100, 200, 400, 600, 800, 1000]
    lams = [10, 20, 40, 80]

    print("Testing jump relations (h-convergence)...")
    print(f"Parameters: k = {lams}, elements = {nelements}")

    from sumpy.expansion.local import (
    AsymptoticDividingLineTaylorExpansion,
    LineTaylorLocalExpansion,
    )
    from sumpy.qbx import LayerPotential

    asymexpn = AsymptoticDividingLineTaylorExpansion(knl, asym_knl, expn_order)
    qbmax_lpot = LayerPotential(
        actx.context, expansion=asymexpn, source_kernels=(knl,), target_kernels=(knl,)
    )

    expn = LineTaylorLocalExpansion(knl, expn_order)
    qbx_lpot = LayerPotential(
        actx.context, expansion=expn, source_kernels=(knl,), target_kernels=(knl,)
    )

    qbx_results = {}
    qbmax_results = {}
    h_values = []

    for lam in lams:
        qbmax_results[lam] = {
            "abs_error": []
        }
        qbx_results[lam] = {
            "abs_error": []
        }
        print(f"Testing for k = {lam}...")
        for nelement in nelements:
            (sources_h, targets_h, centers_in_h, centers_out_h,
            weights_nodes_h, expansion_radii_h,
            normals_h, normals_target_h, hmax_h) = setup_geometry(
                nelement, expn_order, target_order, upsampling_factor
            )

            if lam == lams[0]:
                h_values.append(hmax_h)

            (sources, targets, centers_in, centers_out,
            _, expansion_radii, normal, _) = preprocess_arrays(
                sources_h, targets_h, centers_in_h, centers_out_h,
                weights_nodes_h, expansion_radii_h, normals_h,
                normals_target_h
            )

            sigma = actx.from_numpy(sigma_fun(sources_h) * weights_nodes_h)
            strengths = (sigma,)
            extra_kwargs = {"dsource_vec": normal, "lam": lam}

            # QBMAX
            _, (qbmax_result_lpot1,) = qbmax_lpot(
                actx.queue, targets=targets, sources=sources, centers=centers_in,
                expansion_radii=expansion_radii, strengths=strengths, **extra_kwargs
            )
            _, (qbmax_result_lpot2,) = qbmax_lpot(
                actx.queue, targets=targets, sources=sources, centers=centers_out,
                expansion_radii=expansion_radii, strengths=strengths, **extra_kwargs
            )

            qbmax_result_lpot1 = actx.to_numpy(qbmax_result_lpot1)
            qbmax_result_lpot2 = actx.to_numpy(qbmax_result_lpot2)
            qbmax_error = np.abs(qbmax_result_lpot1 - qbmax_result_lpot2 +
                                                                sigma_fun(targets_h))

            qbmax_results[lam]["abs_error"].append(qbmax_error)

            # QBX
            _, (qbx_result_lpot1,) = qbx_lpot(
                actx.queue, targets=targets, sources=sources, centers=centers_in,
                expansion_radii=expansion_radii, strengths=strengths, **extra_kwargs
            )
            _, (qbx_result_lpot2,) = qbx_lpot(
                actx.queue, targets=targets, sources=sources, centers=centers_out,
                expansion_radii=expansion_radii, strengths=strengths, **extra_kwargs
            )

            qbx_result_lpot1 = actx.to_numpy(qbx_result_lpot1)
            qbx_result_lpot2 = actx.to_numpy(qbx_result_lpot2)
            qbx_error = np.abs(qbx_result_lpot1 - qbx_result_lpot2 +
                                                                sigma_fun(targets_h))

            qbx_results[lam]["abs_error"].append(qbx_error)

    print("Test completed. Visualizing results...")

    def visualize_results(qbmax_results, qbx_results, lams, nelements, target_order,
                        expn_order, upsampling_factor):
        """Plot for jump relation errors (h-convergence)."""
        import matplotlib.pyplot as plt
        from cycler import cycler

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "text.latex.preamble": r"\usepackage{amsmath,amsfonts,lmodern}",
                "pgf.rcfonts": False,
                "axes.prop_cycle": cycler(
                    color=["#002147", "#B22222", "#708090", "#228B22", "#6A0DAD"]
                ),
            }
        )

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))
        qbx_marker = "<"
        qbmax_marker = "o"

        for i, lam in enumerate(lams):
            qbx_errors = [np.max(qbx_results[lam]["abs_error"][j])
                                                    for j in range(len(nelements))]
            qbmax_errors = [np.max(qbmax_results[lam]["abs_error"][j])
                                                    for j in range(len(nelements))]

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 5]

            ax.loglog(h_values, qbx_errors, marker=qbx_marker, linestyle="--",
                    color=color, label=f"QBX, $k={lam}$", markersize=6)
            ax.loglog(h_values, qbmax_errors, marker=qbmax_marker, linestyle="-",
                    color=color, label=f"QBMAX, $k={lam}$", markersize=6)

        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$\ell^{\infty}$ error")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params()

        plt.suptitle(r"Jump Relations of $\mathcal{D}_k\sigma$" + "\n" +
                    rf"$N = {nelements}, p={expn_order}$, "
                    rf"$q={target_order + 1}$, $\kappa={upsampling_factor}$",
                    y=0.98)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0),
                ncol=4, frameon=True, handlelength=2.5)

        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()
    visualize_results(qbmax_results, qbx_results, lams, nelements,
                        target_order, expn_order, upsampling_factor)


if __name__ == "__main__":

    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    from meshmode.array_context import PyOpenCLArrayContext
    actx = PyOpenCLArrayContext(queue)

    main()
