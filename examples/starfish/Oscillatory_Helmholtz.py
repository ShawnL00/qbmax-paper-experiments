"""
QBMAX vs QBX comparison for oscillatory Helmholtz layer potentials on starfish geometry.
"""

import numpy as np


def hankel(k, r):
    """Compute Hankel functions using PyOpenCL."""
    import pyopencl.array as cl_array
    from pyopencl.clmath import hankel_01

    if np.isscalar(r):
        r = np.array([r])
    kr_h = (k * r).astype(np.complex128)
    kr_dev = cl_array.to_device(queue, kr_h)
    h0, h1 = hankel_01(kr_dev)
    return h0, h1


def u_true(k, x):
    """Compute true solution using point sources."""
    result = 0
    for i in range(y.shape[1]):
        r = np.linalg.norm(x - y[:, i])
        h0, _ = hankel(k, r)
        result += h0.get()[0]
    return result


def grad_u_true(k, x):
    """Compute gradient of true solution."""
    result = np.zeros(2, dtype=np.complex128)
    for i in range(y.shape[1]):
        r = np.linalg.norm(x - y[:, i])
        _, h1 = hankel(k, r)
        grad_h0 = -k * h1.get()[0] / r
        result += grad_h0 * (x - y[:, i])
    return result


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
    normals_target_h = actx.to_numpy(flatten(normals_target, actx))\
                                                    .reshape(ambient_dim, -1)

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
    centers_in_h = actx.to_numpy(flatten(centers_in, actx))\
                                            .reshape(ambient_dim, -1)
    centers_out_h = actx.to_numpy(flatten(centers_out, actx))\
                                            .reshape(ambient_dim, -1)
    weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))
    hmax_h = actx.to_numpy(flatten(hmax, actx))

    return (sources_h, targets_h, centers_in_h, centers_out_h, weights_nodes_h,
            expansion_radii_h, normals_h, normals_target_h, hmax_h)


def preprocess(sources, targets, centers_in, centers_out, weights_nodes,
               expansion_radii, normal):
    """Convert numpy arrays to device arrays."""
    return (actx.from_numpy(sources), actx.from_numpy(targets),
            actx.from_numpy(centers_in), actx.from_numpy(centers_out),
            actx.from_numpy(weights_nodes), actx.from_numpy(expansion_radii),
            actx.from_numpy(normal))


def main():
    np.random.seed(42)
    n_sources = 10
    radius = 1.1
    theta = np.random.uniform(0, 2 * np.pi, n_sources)
    global y
    y = radius * (1 + 1/4 * np.sin(5 * theta)) *\
            np.array([np.cos(theta), np.sin(theta)])

    from sumpy.kernel import DirectionalSourceDerivative, HelmholtzKernel

    knl = HelmholtzKernel(2, allow_evanescent=True)
    dknl = DirectionalSourceDerivative(knl, dir_vec_name="dsource_vec")

    # parameters
    nelement = 300
    target_order = 6
    expn_order = 6
    upsampling_factor = 5

    (sources_h, targets_h, centers_in_h, centers_out_h, weights_nodes_h,
        expansion_radii_h, normals_h, normals_target_h, _) = setup_geometry(
        nelement, expn_order, target_order, upsampling_factor
    )

    (sources, targets, centers_in, _, _,
        expansion_radii, normal) = preprocess(
        sources_h, targets_h, centers_in_h, centers_out_h, weights_nodes_h,
        expansion_radii_h, normals_h
    )

    k_values = [10 + 10j, 10 + 20j, 10 + 40j, 10 + 80j,
                20 + 10j, 20 + 20j, 20 + 40j, 20 + 80j,
                40 + 20j, 40 + 40j, 40 + 80j, 40 + 160j,
                80 + 20j, 80 + 40j, 80 + 80j, 80 + 160j]

    taus = [0, 1/8, 1/4, 1/2, 3/4, 7/8, 1]

    s_sigma = {}
    d_sigma = {}
    qbmax_results = {}
    qbx_results = {}
    u_target = {}

    for k in k_values:
        # analytical densities
        s_sigma[k] = np.array([
            grad_u_true(k, sources_h[:, i]).dot(normals_h[:, i])
            for i in range(sources_h.shape[1])
        ])

        d_sigma[k] = np.array([
            u_true(k, sources_h[:, i])
            for i in range(sources_h.shape[1])
        ])

        print(f"Running for k = {k}")

        lam = np.imag(k)
        asym_knl = asym_yukawa(2, lam=lam)

        qbmax_results[k] = {}
        qbx_results[k] = {}
        u_target[k] = {}

        for tau in taus:

            from sumpy.expansion.local import (
                AsymptoticDividingLineTaylorExpansion,
                LineTaylorLocalExpansion,
            )
            from sumpy.qbx import LayerPotential

            asymexpn_S = AsymptoticDividingLineTaylorExpansion(knl, asym_knl,
                                                                expn_order, tau=tau)
            asymexpn_D = AsymptoticDividingLineTaylorExpansion(dknl, asym_knl,
                                                               expn_order, tau=tau)
            lineexpn_S = LineTaylorLocalExpansion(knl, expn_order, tau=tau)
            lineexpn_D = LineTaylorLocalExpansion(dknl, expn_order, tau=tau)

            lplot_S_asym = LayerPotential(actx.context, expansion=asymexpn_S,
                                        source_kernels=(knl,), target_kernels=(knl,))
            lplot_D_asym = LayerPotential(actx.context, expansion=asymexpn_D,
                                        source_kernels=(dknl,), target_kernels=(dknl,))
            lplot_S_line = LayerPotential(actx.context, expansion=lineexpn_S,
                                        source_kernels=(knl,), target_kernels=(knl,))
            lplot_D_line = LayerPotential(actx.context, expansion=lineexpn_D,
                                        source_kernels=(dknl,), target_kernels=(dknl,))

            # strengths
            sigma_S = s_sigma[k] * weights_nodes_h
            strength_S = (actx.from_numpy(sigma_S),)
            sigma_D = d_sigma[k] * weights_nodes_h
            strength_D = (actx.from_numpy(sigma_D),)

            # Evaluate single-layer potentials
            S_extra_kwargs = {"k": k}
            _, (qbmax_S,) = lplot_S_asym(actx.queue, targets=targets, sources=sources,
                                        centers=centers_in, strengths=strength_S,
                                        expansion_radii=expansion_radii,
                                        **S_extra_kwargs)

            _, (qbx_S,) = lplot_S_line(actx.queue, targets=targets, sources=sources,
                                        centers=centers_in, strengths=strength_S,
                                        expansion_radii=expansion_radii,
                                        **S_extra_kwargs)

            # Evaluate double-layer potentials
            D_extra_kwargs = {"k": k, "dsource_vec": normal}
            _, (qbmax_D,) = lplot_D_asym(actx.queue, targets=targets, sources=sources,
                                        centers=centers_in, strengths=strength_D,
                                        expansion_radii=expansion_radii,
                                        **D_extra_kwargs)

            _, (qbx_D,) = lplot_D_line(actx.queue, targets=targets, sources=sources,
                                        centers=centers_in, strengths=strength_D,
                                        expansion_radii=expansion_radii,
                                        **D_extra_kwargs)

            qbmax_S = actx.to_numpy(qbmax_S) * \
                np.exp(-lam * (1 - tau) * expansion_radii_h)
            qbmax_D = actx.to_numpy(qbmax_D) * \
                np.exp(-lam * (1 - tau) * expansion_radii_h)
            qbx_S = actx.to_numpy(qbx_S)
            qbx_D = actx.to_numpy(qbx_D)

            u_target_tau = targets_h - (1 - tau) * normals_target_h * expansion_radii_h
            u_target[k][tau] = np.array([
                u_true(k, u_target_tau[:, i]) for i in range(targets_h.shape[1])
            ])

            qbmax_error = qbmax_D - qbmax_S + u_target[k][tau]
            qbx_error = qbx_D - qbx_S + u_target[k][tau]

            qbmax_results[k][tau] = {"error": qbmax_error}
            qbx_results[k][tau] = {"error": qbx_error}

    print("Visualizing results...")

    def visualize_results():
        import matplotlib.pyplot as plt
        from cycler import cycler
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath,amsfonts,lmodern}",
                "pgf.rcfonts": False,
                "pgf.texsystem": "pdflatex",
                "axes.prop_cycle": cycler(
                    color=["#002147", "#B22222", "#708090", "#228B22", "#6A0DAD"]
                ),
            }
        )

        k_groups = {}
        for k in k_values:
            real_k = k.real
            if real_k not in k_groups:
                k_groups[real_k] = []
            k_groups[real_k].append(k)

        sorted_real_k = sorted(k_groups.keys())

        n_real_k = len(sorted_real_k)
        _, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        axs = axs.flatten()

        qbx_marker = ">"
        qbmax_marker = "o"
        taus_plot = taus[1:]

        for subplot_idx, real_k in enumerate(sorted_real_k):
            if subplot_idx >= len(axs):
                break

            ax = axs[subplot_idx]

            for color_idx, k in enumerate(k_groups[real_k]):
                qbx_errs = []
                qbmax_errs = []

                for tau in taus_plot:
                    if k in qbmax_results and tau in qbmax_results[k]:
                        qbmax_error = qbmax_results[k][tau]["error"]
                        qbx_error = qbx_results[k][tau]["error"]

                        qbmax_err = np.linalg.norm(qbmax_error, ord=np.inf)
                        qbx_err = np.linalg.norm(qbx_error, ord=np.inf)

                        qbmax_errs.append(qbmax_err)
                        qbx_errs.append(qbx_err)

                color = plt.rcParams["axes.prop_cycle"].by_key()["color"][color_idx % 5]

                imag_k = k.imag

                ax.plot(
                    np.log10(taus_plot),
                    np.log10(qbx_errs),
                    marker=qbx_marker,
                    linestyle="--",
                    color=color,
                    label=f"QBX, $\\mathrm{{Im}}(k)={imag_k:g}$",
                )
                ax.plot(
                    np.log10(taus_plot),
                    np.log10(qbmax_errs),
                    marker=qbmax_marker,
                    linestyle="-",
                    color=color,
                    label=f"QBMAX, $\\mathrm{{Im}}(k)={imag_k:g}$",
                )

            ax.set_title(f"$\\mathrm{{Re}}(k) = {real_k:g}$")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)

        for i in range(n_real_k):
            if i >= 2:
                axs[i].set_xlabel(r"$\log_{10}(\tau)$")
            if i % 2 == 0:
                axs[i].set_ylabel(r"$\log_{10}(\mathrm{err}(\tau))$")

        for i in range(n_real_k, len(axs)):
            axs[i].set_visible(False)

        plt.suptitle(
            rf"$N={nelement}$, $p={expn_order}$, $q={target_order+1}$, "
            rf"$\kappa={upsampling_factor}$",
            fontsize=14,
            y=0.98,
            x=0.52,
        )
        plt.tight_layout()
        plt.show()
    visualize_results()


if __name__ == "__main__":
    import pyopencl as cl
    from meshmode.array_context import PyOpenCLArrayContext

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    main()
