"""
Neumann BVP Problem
    -Δu + k²u = 0   in Ω
    ∂ᵥu = f         on ∂Ω
Starfish domain: (x(θ), y(θ)) = r(θ)(cos(θ), sin(θ)) where r(θ) = 1 + (1/4)sin(5θ)
Reference domain: scaled by factor 1.1 for boundary condition generation
The reference solution u_ref = S_k[σ̃] uses density σ̃(y) = cos(10θ)sin(5θ) + 2
where θ = atan2(y₂,y₁). The Neumann data f = ∂ᵥu_ref|_{∂Ω}.
Solution u = S_k[σ] leads to the second-kind integral equation:
    (1/2)σ(x) + S'_k[σ](x) = f(x), x ∈ ∂Ω
"""


def visualize_results(data):
    import matplotlib.pyplot as plt
    from cycler import cycler
    from scipy import stats
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts,lmodern}",
        "pgf.rcfonts": False,
        "axes.prop_cycle": cycler(
            color=["#002147", "#B22222", "#708090", "#228B22", "#6A0DAD"]
        ),
    })

    results = data["results"]
    params = data["parameters"]
    lams = params["lam_values"]
    taus = params["tau_values"]
    nelements = params["nelement_values"]
    hmax_values = params["hmax_values"]
    expn_order = params["expn_order"]
    target_order = params["target_order"]
    upsampling_factor = params["upsampling_factor"]

    n_lam = len(lams)
    _fig, axes = plt.subplots(2, n_lam, figsize=(10, 5.5), sharex=False, sharey="row")

    for i, lam in enumerate(lams):
        ax_error = axes[0, i]

        for j, tau in enumerate(taus):
            relative_errors = []
            hmax_list = []
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][j % 5]

            for n in nelements:
                rel_error = results[lam][tau][n]["error"]  / \
                                                np.max(results[lam][tau][n]["ref"])
                relative_errors.append(rel_error)
                hmax_list.append(hmax_values[n])

            ax_error.loglog(hmax_list, relative_errors, "o-",
                        color=color, linewidth=2, markersize=6,
                        label=f"$\\tau = {tau}$")

        fit_tau = 1
        hmax_fit = [hmax_values[n] for n in nelements[1:6]]
        error_fit = [np.abs(results[lam][fit_tau][n]["error"] /
                                        np.max(results[lam][fit_tau][n]["ref"]))
                    for n in nelements[1:6]]

        log_hmax = np.log10(hmax_fit)
        log_error = np.log10(error_fit)
        slope, intercept, _, _, _ = stats.linregress(log_hmax, log_error)

        hmax_range = np.logspace(np.log10(min(hmax_list)), np.log10(max(hmax_list)), 50)
        fitted_line = 10**(slope * np.log10(hmax_range) + intercept)

        ax_error.loglog(hmax_range, fitted_line, "k--",
                    linewidth=2, alpha=0.8,
                    label=f"$\\mathcal{{O}}(h^{{{slope:.1f}}})$")

        ax_error.set_xlabel(r"$h$")
        ax_error.set_ylabel(r"relative error $(\tau)$, $\ell^{\infty}$")
        ax_error.set_title(f"$k = {lam}$ - $h$-convergence")
        ax_error.grid(True, alpha=0.3)
        ax_error.legend(loc="lower right")

        ax_gmres = axes[1, i]

        gmres_tau = taus[0]

        if lam in results and gmres_tau in results[lam]:
            gmres_iters = []

            for n in nelements:
                gmres_iters.append(results[lam][gmres_tau][n]["gmres_iters"])

            ax_gmres.plot(nelements, gmres_iters, "ko-",
                        linewidth=2, markersize=6)

        ax_gmres.set_xlabel(r"$N$")
        ax_gmres.set_ylabel("GMRES iterations")
        ax_gmres.set_title(f"$k = {lam}$ - GMRES vs $N$")
        ax_gmres.grid(True, alpha=0.3)
    plt.suptitle(f"Interior Neumann Problem - $p={expn_order}$, "
                 f"$q={target_order + 1}$, $\\kappa={upsampling_factor}$")
    plt.tight_layout()
    plt.show()


def starfish_parametrization(t, n_arms=5, amplitude=0.25):
    """Parametrization of the starfish domain."""
    theta = 2 * np.pi * t

    r = 1 + amplitude * np.sin(n_arms * theta)
    dr_dt = amplitude * n_arms * 2 * np.pi * np.cos(n_arms * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    dx_dt = dr_dt * np.cos(theta) - r * np.sin(theta) * 2 * np.pi
    dy_dt = dr_dt * np.sin(theta) + r * np.cos(theta) * 2 * np.pi

    jacobian_norm = np.sqrt(dx_dt**2 + dy_dt**2)

    tangent_x = dx_dt / jacobian_norm
    tangent_y = dy_dt / jacobian_norm

    normal_x = tangent_y
    normal_y = -tangent_x

    coords = np.vstack([x, y])
    tangents = np.vstack([tangent_x, tangent_y])
    normals = np.vstack([normal_x, normal_y])

    return coords, tangents, normals, jacobian_norm


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


def resampling_matrix(target_discr, source_discr):
    """Upsampling matrix from target to source discretization."""
    import modepy as mp
    return mp.resampling_matrix(
        target_discr.groups[0].basis_obj().functions,
        source_discr.groups[0].unit_nodes,
        target_discr.groups[0].unit_nodes
    )


def setup_geometry(nelement, expn_order, target_order, upsampling_factor=1):
    """Set up mesh, discretization, and QBX source."""
    mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelement + 1), target_order)
    pre_density_discr = Discretization(
        actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order)
    )

    qbx = QBXLayerPotentialSource(
        pre_density_discr,
        upsampling_factor * target_order,
        expn_order,
        fmm_order=False
    )
    places = GeometryCollection({"qbx": qbx}, auto_where=("qbx"))

    target_discr = places.get_discretization("qbx", sym.QBX_SOURCE_STAGE1)
    source_discr = places.get_discretization("qbx", sym.QBX_SOURCE_QUAD_STAGE2)

    ambient_dim = 2

    h_max = actx.to_numpy(bind(places, sym.h_max(qbx.ambient_dim))(actx))

    targets = target_discr.nodes()
    targets_h = actx.to_numpy(flatten(targets, actx)).reshape(ambient_dim, -1)
    targets = actx.from_numpy(targets_h)

    sources = source_discr.nodes()
    sources_h = actx.to_numpy(flatten(sources, actx)).reshape(ambient_dim, -1)
    sources = actx.from_numpy(sources_h)

    dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_QUAD_STAGE2)
    target_dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_STAGE1)

    normals = bind(places, sym.normal(qbx.ambient_dim, dofdesc=dofdesc))(actx)
    normals = normals.as_vector(object)
    normals_h = actx.to_numpy(flatten(normals, actx)).reshape(ambient_dim, -1)
    normals = actx.from_numpy(normals_h)

    targets_normal = bind(places,
                          sym.normal(qbx.ambient_dim, dofdesc=target_dofdesc))(actx)
    targets_normal = targets_normal.as_vector(object)
    targets_normals_h = actx.to_numpy(flatten(targets_normal, actx))
    targets_normals_h = targets_normals_h.reshape(ambient_dim, -1)

    expansion_radii = bind(places, sym.expansion_radii(ambient_dim))(actx)
    expansion_radii_h = actx.to_numpy(flatten(expansion_radii, actx))
    expansion_radii = actx.from_numpy(expansion_radii_h)

    centers_in = bind(places, sym.expansion_centers(qbx.ambient_dim, -1))(actx)
    centers_in_h = actx.to_numpy(flatten(centers_in, actx)).reshape(ambient_dim, -1)
    centers_in = actx.from_numpy(centers_in_h)

    weights_nodes = bind(places, sym.weights_and_area_elements(
        ambient_dim=2, dim=1, dofdesc=dofdesc))(actx)
    weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))

    p_submat = resampling_matrix(target_discr, source_discr)

    return (targets, sources, normals, expansion_radii, centers_in,
            targets_h, sources_h, normals_h, targets_normals_h, expansion_radii_h,
            weights_nodes_h, p_submat, target_discr, source_discr, h_max)


def generate_boundary_condition(lam, targets, normals_target_h, tolerance=1e-13):
    """Generate boundary condition for the Neumann problem."""
    from meshmode.mesh.processing import affine_map
    from pytential.target import PointsTarget
    from sumpy.kernel import YukawaKernel

    target_order_ref = 10
    expn_order_ref = 8
    upsampling_factor_ref = 4

    ref_element = [200, 400, 800, 1500, 2500]
    previous_result = None

    base_knl = YukawaKernel(2)
    kernel_kwargs = {"lam": lam}

    def op0(**kwargs):
        kwargs.update(kernel_kwargs)
        return sym.d_dx(2, sym.S(base_knl, sym.var("sigma"), **kwargs))

    def op1(**kwargs):
        kwargs.update(kernel_kwargs)
        return sym.d_dy(2, sym.S(base_knl, sym.var("sigma"), **kwargs))

    print(f"Generating boundary condition: k={lam}")
    for nelement in ref_element:
        base_mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelement + 1),
                                    target_order_ref)
        scaling = 1.1
        mesh_ref = affine_map(base_mesh, A=np.array([[scaling, 0], [0, scaling]]))

        pre_density_discr_ref = Discretization(
            actx, mesh_ref, InterpolatoryQuadratureSimplexGroupFactory(target_order_ref)
        )

        qbx_ref = QBXLayerPotentialSource(
            pre_density_discr_ref,
            upsampling_factor_ref * target_order_ref,
            expn_order_ref,
            fmm_order=False,
            target_association_tolerance=0.05
        )

        targets_ref = targets
        places_ref = GeometryCollection({
            "qbx": qbx_ref,
            "targets": PointsTarget(targets_ref),
        }, auto_where=("qbx", "targets"))

        source_discr_ref = places_ref.get_discretization("qbx")
        sources_ref_thawed = actx.thaw(source_discr_ref.nodes())
        angle = actx.np.arctan2(sources_ref_thawed[1], sources_ref_thawed[0])
        sigma = actx.np.cos(10 * angle) * actx.np.sin(5 * angle) + 2

        result_x = actx.to_numpy(
            bind(places_ref, op0(
                source="qbx",
                target="targets",
                qbx_forced_limit=-2))(actx, sigma=sigma, lam=lam)
        )

        result_y = actx.to_numpy(
            bind(places_ref, op1(
                source="qbx",
                target="targets",
                qbx_forced_limit=-2))(actx, sigma=sigma, lam=lam)
        )

        current_result = (result_x * normals_target_h[0] +
                        result_y * normals_target_h[1])
        current_result = current_result.flatten()

        if previous_result is not None:
            linf_abs_diff = np.linalg.norm(np.abs(current_result - previous_result),
                                           ord=np.inf)

            if linf_abs_diff < tolerance:
                return current_result

        previous_result = current_result.copy()

    print(f"    Warning: Used finest mesh: nelement={ref_element[-1]}")
    return current_result


def solve_neumann_system(geometry_data, lam, nelement, target_order, upsampling_factor,
                         expn_order):
    """Nyström method with GMRES"""
    (targets, sources, _normals, expansion_radii, centers_in,
    _targets_h, _sources_h, _normals_h, targets_normals_h, _expansion_radii_h,
    weights_nodes_h, p_submat, target_discr, _source_discr, _hmax) = geometry_data

    rhs = generate_boundary_condition(lam, targets, targets_normals_h)
    rhs = rhs.flatten()

    from sumpy.kernel import YukawaKernel
    base_knl = YukawaKernel(2)
    asym_knl = asym_yukawa(2)

    from sumpy.expansion.local import AsymptoticDividingLineTaylorExpansion
    asymexpn = AsymptoticDividingLineTaylorExpansion(base_knl, asym_knl, expn_order)

    extra_kwargs = {"lam": lam}

    from sumpy.kernel import AxisTargetDerivative
    from sumpy.qbx import LayerPotentialMatrixGenerator
    mat_gen0 = LayerPotentialMatrixGenerator(
        actx.context,
        expansion=asymexpn,
        source_kernels=(base_knl,),
        target_kernels=(AxisTargetDerivative(0, base_knl),)
    )

    mat_gen1 = LayerPotentialMatrixGenerator(
        actx.context,
        expansion=asymexpn,
        source_kernels=(base_knl,),
        target_kernels=(AxisTargetDerivative(1, base_knl),)
    )

    _, (mat_in0,) = mat_gen0(
        actx.queue,
        targets=targets,
        sources=sources,
        expansion_radii=expansion_radii,
        centers=centers_in,
        **extra_kwargs
    )

    _, (mat_in1,) = mat_gen1(
        actx.queue,
        targets=targets,
        sources=sources,
        expansion_radii=expansion_radii,
        centers=centers_in,
        **extra_kwargs
    )

    mat_in0 = actx.to_numpy(mat_in0)
    weighted_mat_in0 = mat_in0 * weights_nodes_h[None, :]

    mat_in1 = actx.to_numpy(mat_in1)
    weighted_mat_in1 = mat_in1 * weights_nodes_h[None, :]

    mat = (weighted_mat_in0 * targets_normals_h[0][:, None] +
        weighted_mat_in1 * targets_normals_h[1][:, None])

    mat = mat.reshape(
        target_discr.ndofs, nelement, upsampling_factor * target_order + 1
    ).transpose(1, 0, 2) @ p_submat
    mat = mat.transpose(1, 0, 2).reshape(
        target_discr.ndofs, nelement * (target_order + 1)
    )

    iter_count = [0]

    def gmres_callback(rk):
        iter_count[0] += 1

    from scipy.sparse.linalg import gmres
    density_c, info = gmres(mat, rhs, rtol=1e-14, callback=gmres_callback)

    if info != 0:
        raise RuntimeError(f"GMRES did not converge, info={info}")
    else:
        print(f"  GMRES converged in {iter_count[0]} iterations")

    density_f = (density_c.reshape(nelement, -1) @ p_submat.T).reshape(-1)

    return density_f, iter_count[0]


def evaluate_solution(density_f, geometry_data, eval_targets_h, eval_targets_normals_h,
                    eval_expansion_radii_h, lam, tau, expn_order):
    """Evaluate solution at target points after obtaining the discrete density."""
    (_targets, sources, _normals, _expansion_radii, _centers_in,
    _targets_h, _sources_h, _normals_h, _targets_normals_h, _expansion_radii_h,
    weights_nodes_h, _p_submat, _target_discr, _source_discr, _hmax) = geometry_data

    eval_targets = actx.from_numpy(eval_targets_h)
    eval_centers_in = actx.from_numpy(eval_targets_h -
                                    eval_expansion_radii_h * eval_targets_normals_h)

    from sumpy.kernel import YukawaKernel
    base_knl = YukawaKernel(2)
    asym_knl = asym_yukawa(2)

    from sumpy.expansion.local import AsymptoticDividingLineTaylorExpansion
    asymexpn = AsymptoticDividingLineTaylorExpansion(base_knl, asym_knl, expn_order,
                                                     tau=tau)

    extra_kwargs = {"lam": lam}

    from sumpy.qbx import LayerPotentialMatrixGenerator
    mat_gen = LayerPotentialMatrixGenerator(
        actx.context,
        expansion=asymexpn,
        source_kernels=(base_knl,),
        target_kernels=(base_knl,)
    )

    _, (eval_mat,) = mat_gen(
        actx.queue,
        targets=eval_targets,
        sources=sources,
        expansion_radii=eval_expansion_radii_h,
        centers=eval_centers_in,
        **extra_kwargs
    )

    eval_mat = actx.to_numpy(eval_mat)
    eval_mat = eval_mat * weights_nodes_h[None, :]

    eval_result = eval_mat @ density_f *\
                                np.exp(-lam * eval_expansion_radii_h * (1 - tau))

    return eval_result


def generate_reference_solution(lam, tau, targets_h, side, expansion_radii_h,
                                normals_target_h, tolerance=1e-13):
    """Generate reference solution at target points."""
    targets_ref = actx.from_numpy(
        targets_h + side/2 * (1-tau) * expansion_radii_h * normals_target_h
    )

    target_order_ref = 10
    expn_order_ref = 8
    upsampling_factor_ref = 4

    ref_element = [200, 400, 800, 1500, 2000]
    previous_result = None
    from sumpy.kernel import YukawaKernel
    base_knl = YukawaKernel(2)
    kernel_kwargs = {"lam": lam}

    def op(**kwargs):
        kwargs.update(kernel_kwargs)
        return sym.S(base_knl, sym.var("sigma"), **kwargs)

    for nelement in ref_element:
        base_mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelement + 1),
                                    target_order_ref)
        scaling = 1.1
        from meshmode.mesh.processing import affine_map
        mesh_ref = affine_map(base_mesh, A=np.array([[scaling, 0], [0, scaling]]))

        pre_density_discr_ref = Discretization(
            actx, mesh_ref, InterpolatoryQuadratureSimplexGroupFactory(target_order_ref)
        )

        qbx_ref = QBXLayerPotentialSource(
            pre_density_discr_ref,
            upsampling_factor_ref * target_order_ref,
            expn_order_ref,
            fmm_order=False
        )

        from pytential.target import PointsTarget
        places_ref = GeometryCollection({
            "qbx": qbx_ref,
            "targets": PointsTarget(targets_ref),
        }, auto_where=("qbx", "targets"))

        source_discr_ref = places_ref.get_discretization("qbx")
        sources_ref_thawed = actx.thaw(source_discr_ref.nodes())
        angle = actx.np.arctan2(sources_ref_thawed[1], sources_ref_thawed[0])
        sigma = actx.np.cos(10 * angle) * actx.np.sin(5 * angle) + 2

        current_result = actx.to_numpy(
            bind(places_ref, op(
                source="qbx",
                target="targets",
                qbx_forced_limit=side))(actx, sigma=sigma, lam=lam)
        )

        if previous_result is not None:
            linf_abs_diff = np.linalg.norm(np.abs(current_result - previous_result),
                                           ord=np.inf)

            if linf_abs_diff < tolerance:
                return current_result

        previous_result = current_result.copy()

    print(f"    Warning: Used finest mesh: nelement={ref_element[-1]}")
    return current_result


def main():
    # Parameters
    target_order = 6
    expn_order = 6
    upsampling_factor = 5

    lams = [10, 20, 40, 80]
    taus = [0, 0.5, 1.0]
    nelements = [100, 200, 300, 600, 800, 1000, 1200, 1400, 1600, 1800]
    n_eval_points = 1000

    print(f"\n{'='*60}")
    print("QBMAX NEUMANN BVP")
    print(f"{'='*60}")

    t = np.linspace(0, 1, n_eval_points, endpoint=False)
    eval_targets_h, _, eval_targets_normals_h, jac = starfish_parametrization(
        t, n_arms=5, amplitude=0.25
    )

    results = {}
    hmax_values = {}

    for lam in lams:
        results[lam] = {}
        for tau in taus:
            results[lam][tau] = {}

    for nelement in nelements:
        geometry_data = setup_geometry(nelement, expn_order, target_order,
                                       upsampling_factor)
        eval_expansion_radii_h = jac / (2 * nelement)

        hmax = geometry_data[-1]
        hmax_values[nelement] = hmax

        for lam in lams:
            density, gmres_iters = solve_neumann_system(geometry_data, lam,
                                                        nelement, target_order,
                                                        upsampling_factor,
                                                        expn_order)
            for tau in taus:
                eval_result = evaluate_solution(
                    density, geometry_data, eval_targets_h, eval_targets_normals_h,
                    eval_expansion_radii_h, lam, tau, expn_order
                )

                ref_solution = generate_reference_solution(
                    lam, tau, eval_targets_h, -2, eval_expansion_radii_h,
                    eval_targets_normals_h
                )

                error = np.max(np.abs(eval_result - ref_solution))

                results[lam][tau][nelement] = {
                    "eval": eval_result,
                    "ref": ref_solution,
                    "error": error,
                    "gmres_iters": gmres_iters
                }

            print()

    results = {
        "results": results,
        "parameters": {
            "lam_values": lams,
            "tau_values": taus,
            "nelement_values": nelements,
            "hmax_values": hmax_values,
            "target_order": target_order,
            "expn_order": expn_order,
            "upsampling_factor": upsampling_factor,
            "n_eval_points": n_eval_points
        }
    }

    with open(file_name, "wb") as f:
        pickle.dump(results, f)

    print("\nComputation completed and results saved! Visualizing results...\n")
    visualize_results(results)


if __name__ == "__main__":
    import os
    import pickle
    from pathlib import Path

    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    file_name = "Neumann_bvp.pkl"
    file_exists = os.path.exists(file_name)

    import numpy as np
    if file_exists:
        print(f"Loading and plotting data from {file_name}...")
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            visualize_results(data)
    else:
        print(f"{file_name} not found. Running computation...")
        import pyopencl as cl
        from meshmode.array_context import PyOpenCLArrayContext

        cl_ctx = cl.create_some_context()
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue)

        from arraycontext import flatten
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory,
        )
        from meshmode.mesh.generation import make_curve_mesh, starfish
        from pytential import GeometryCollection, bind, sym
        from pytential.qbx import QBXLayerPotentialSource

        main()
