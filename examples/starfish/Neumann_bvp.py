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

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import gmres

import pyopencl as cl
from arraycontext import flatten
from pytential.array_context import PyOpenCLArrayContext

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import InterpolatoryQuadratureSimplexGroupFactory
from meshmode.mesh.generation import make_curve_mesh, starfish
from meshmode.mesh.processing import affine_map
import modepy as mp

from pytential.qbx import QBXLayerPotentialSource
from pytential import GeometryCollection, bind, sym
from pytential.target import PointsTarget

from sumpy.kernel import YukawaKernel, AxisTargetDerivative
from sumpy.expansion.local import (
    AsymptoticDividingLineTaylorExpansion,
    LineTaylorLocalExpansion,
)
from sumpy.qbx import LayerPotentialMatrixGenerator


cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)
actx = PyOpenCLArrayContext(queue)


def resampling_matrix(target_discr, source_discr):
    return mp.resampling_matrix(
        target_discr.groups[0].basis_obj().functions,
        source_discr.groups[0].unit_nodes, 
        target_discr.groups[0].unit_nodes
    )


def asymptotic_yukawa_kernel(dim, lam=None):
    from pymbolic import primitives, var
    from sumpy.symbolic import pymbolic_real_norm_2, SpatialConstant
    
    b = pymbolic_real_norm_2(primitives.make_sym_vector("b", dim))
    
    if lam is not None:
        expr = var("exp")(-lam * b * (1 - var('tau'))) 
    else:
        lam = SpatialConstant("lam")
        expr = var("exp")(-lam * b * (1 - var('tau')))
    
    return expr


def make_local_expansion(base_knl, asym_knl, expn_order, method, tau=1):
    if method == "qbmax":
        return AsymptoticDividingLineTaylorExpansion(
            base_knl, expn_order, asymptotic=asym_knl, tau=tau)
    if method == "qbx":
        return LineTaylorLocalExpansion(base_knl, expn_order, tau=tau)
    raise ValueError(f"unknown expansion method '{method}'")


def starfish_parametrization(t, n_arms=5, amplitude=0.25):
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


def setup_geometry(nelement, expn_order, target_order, upsampling_factor):
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
    places = GeometryCollection({"qbx": qbx}, auto_where=('qbx'))

    target_discr = places.get_discretization('qbx', sym.QBX_SOURCE_STAGE1)
    source_discr = places.get_discretization('qbx', sym.QBX_SOURCE_QUAD_STAGE2)

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

    normals = bind(places, sym.normal(qbx.ambient_dim, dofdesc=dofdesc))(actx).as_vector(object)
    normals_h = actx.to_numpy(flatten(normals, actx)).reshape(ambient_dim, -1)
    normals = actx.from_numpy(normals_h)
    
    targets_normal = bind(places, sym.normal(qbx.ambient_dim, dofdesc=target_dofdesc))(actx).as_vector(object)
    targets_normals_h = actx.to_numpy(flatten(targets_normal, actx)).reshape(ambient_dim, -1)

    expansion_radii = bind(places, sym.expansion_radii(ambient_dim))(actx)
    expansion_radii_h = actx.to_numpy(flatten(expansion_radii, actx))
    expansion_radii = actx.from_numpy(expansion_radii_h)

    centers_in = bind(places, sym.expansion_centers(qbx.ambient_dim, -1))(actx)
    centers_in_h = actx.to_numpy(flatten(centers_in, actx)).reshape(ambient_dim, -1)
    centers_in = actx.from_numpy(centers_in_h)

    weights_nodes = bind(places, sym.weights_and_area_elements(
        ambient_dim=2, dim=1, dofdesc=dofdesc))(actx)
    weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))

    P_submat = resampling_matrix(target_discr, source_discr)
    
    return (targets, sources, normals, expansion_radii, centers_in,
            targets_h, sources_h, normals_h, targets_normals_h, expansion_radii_h,
            weights_nodes_h, P_submat, target_discr, source_discr, h_max)


def generate_boundary_condition(lam, tau, targets_h, side, expansion_radii_h, normals_target_h, tolerance=1e-13):
    targets_ref = actx.from_numpy(
        targets_h + side/2 * (1-tau) * expansion_radii_h * normals_target_h
    )
    
    target_order_ref = 10
    expn_order_ref = 8
    upsampling_factor_ref = 4
    
    ref_element = [100, 200, 400, 800, 1000, 1500, 2000, 2500]
    previous_result = None
    
    print(f'Generating boundary condition: k={lam}, tau={tau}')
    
    for nelement in ref_element:
        base_mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelement + 1), target_order_ref)
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
        
        places_ref = GeometryCollection({
            "qbx": qbx_ref,
            "targets": PointsTarget(targets_ref),
        }, auto_where=("qbx", "targets"))
        
        base_knl = YukawaKernel(2)
        kernel_kwargs = {"lam": lam}
        
        def op0(**kwargs):
            kwargs.update(kernel_kwargs)
            return sym.d_dx(2, sym.S(base_knl, sym.var("sigma"), **kwargs))
        
        def op1(**kwargs):
            kwargs.update(kernel_kwargs)
            return sym.d_dy(2, sym.S(base_knl, sym.var("sigma"), **kwargs))
        
        source_discr_ref = places_ref.get_discretization('qbx')
        sources_ref_thawed = actx.thaw(source_discr_ref.nodes())
        angle = actx.np.arctan2(sources_ref_thawed[1], sources_ref_thawed[0])
        sigma = actx.np.cos(10 * angle) * actx.np.sin(5 * angle) + 2
        
        result_x = actx.to_numpy(
            bind(places_ref, op0(
                source="qbx",
                target="targets",
                qbx_forced_limit=side))(actx, sigma=sigma, lam=lam)
        )
        
        result_y = actx.to_numpy(
            bind(places_ref, op1(
                source="qbx",
                target="targets",
                qbx_forced_limit=side))(actx, sigma=sigma, lam=lam)
        )
        
        current_result = (result_x * normals_target_h[0] + 
                         result_y * normals_target_h[1])
        current_result = current_result.flatten()
        
        if previous_result is not None:
            linf_abs_diff = np.linalg.norm(np.abs(current_result - previous_result), ord=np.inf)
            
            if linf_abs_diff < tolerance:
                print(f"    Converged with nelement={nelement}, diff={linf_abs_diff:.2e}")
                return current_result
        
        previous_result = current_result.copy()
    
    print(f"    Warning: Used finest mesh: nelement={ref_element[-1]}")
    return current_result


def solve_neumann_system(
        geometry_data, lam, nelement, target_order, upsampling_factor,
        expn_order, method):
    
    (targets, sources, normals, expansion_radii, centers_in,
     targets_h, sources_h, normals_h, targets_normals_h, expansion_radii_h,
     weights_nodes_h, P_submat, target_discr, source_discr, hmax) = geometry_data
            
    rhs = generate_boundary_condition(lam, 1, targets_h, -2, expansion_radii_h, targets_normals_h)
    rhs = rhs.flatten()

    base_knl = YukawaKernel(2)
    asym_knl = asymptotic_yukawa_kernel(2)
    expn = make_local_expansion(base_knl, asym_knl, expn_order, method)
    
    extra_kwargs = {"lam": lam}
     
    mat_gen0 = LayerPotentialMatrixGenerator(
        expansion=expn,
        source_kernels=(base_knl,),
        target_kernels=(AxisTargetDerivative(0, base_knl),)
    )

    mat_gen1 = LayerPotentialMatrixGenerator(
        expansion=expn,
        source_kernels=(base_knl,),
        target_kernels=(AxisTargetDerivative(1, base_knl),)
    )

    (mat_in0,) = mat_gen0(
        actx,
        targets=targets,
        sources=sources,
        expansion_radii=expansion_radii,
        centers=centers_in,
        **extra_kwargs
    )

    (mat_in1,) = mat_gen1(
        actx,
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
    ).transpose(1, 0, 2) @ P_submat
    mat = mat.transpose(1, 0, 2).reshape(
        target_discr.ndofs, nelement * (target_order + 1)
    )

    iter_count = [0]
    def gmres_callback(_residual):
        iter_count[0] += 1

    density_c, info = gmres(
        mat, rhs, rtol=1e-12, callback=gmres_callback,
        callback_type="legacy")

    if info != 0:
        raise RuntimeError(f"GMRES did not converge, info={info}")
    else:
        print(f"  GMRES converged in {iter_count[0]} iterations")

    density_f = (density_c.reshape(nelement, -1) @ P_submat.T).reshape(-1) 
    
    return density_f, iter_count[0]


def evaluate_solution(density_f, geometry_data, eval_targets_h, eval_targets_normals_h, 
                     eval_expansion_radii_h, lam, tau, expn_order, method):
    (targets, sources, normals, expansion_radii, centers_in,
     targets_h, sources_h, normals_h, targets_normals_h, expansion_radii_h,
     weights_nodes_h, P_submat, target_discr, source_discr, hmax) = geometry_data

    eval_targets = actx.from_numpy(eval_targets_h)
    eval_centers_in = actx.from_numpy(eval_targets_h - eval_expansion_radii_h * eval_targets_normals_h)
    eval_expansion_radii = actx.from_numpy(eval_expansion_radii_h)

    base_knl = YukawaKernel(2)
    asym_knl = asymptotic_yukawa_kernel(2)
    expn = make_local_expansion(base_knl, asym_knl, expn_order, method, tau=tau)
    
    extra_kwargs = {"lam": lam}

    mat_gen = LayerPotentialMatrixGenerator(
        expansion=expn,
        source_kernels=(base_knl,),
        target_kernels=(base_knl,)
    )

    (eval_mat,) = mat_gen(
        actx,
        targets=eval_targets,
        sources=sources,
        expansion_radii=eval_expansion_radii,
        centers=eval_centers_in,
        **extra_kwargs
    )

    eval_mat = actx.to_numpy(eval_mat) 
    eval_mat = eval_mat * weights_nodes_h[None, :]
    
    eval_result = eval_mat @ density_f
    if method == "qbmax":
        eval_result = eval_result * np.exp(-lam * eval_expansion_radii_h * (1 - tau))

    return eval_result


def generate_reference_solution(lam, tau, targets_h, side, expansion_radii_h, normals_target_h, tolerance=1e-13):
    targets_ref = actx.from_numpy(
        targets_h + side/2 * (1-tau) * expansion_radii_h * normals_target_h
    )
    
    target_order_ref = 10
    expn_order_ref = 8
    upsampling_factor_ref = 4
    
    ref_element = [100, 200, 400, 800, 1000, 1500, 2000, 2500]
    previous_result = None
    
    print(f"  Generating reference: k={lam}, tau={tau}")
    
    for nelement in ref_element:
        base_mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelement + 1), target_order_ref)
        scaling = 1.1
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
        
        places_ref = GeometryCollection({
            "qbx": qbx_ref,
            "targets": PointsTarget(targets_ref),
        }, auto_where=("qbx", "targets"))
        
        base_knl = YukawaKernel(2)
        kernel_kwargs = {"lam": lam}
        
        def op(**kwargs):
            kwargs.update(kernel_kwargs)
            return sym.S(base_knl, sym.var("sigma"), **kwargs)
        
        source_discr_ref = places_ref.get_discretization('qbx')
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
            linf_abs_diff = np.linalg.norm(np.abs(current_result - previous_result), ord=np.inf)
            
            if linf_abs_diff < tolerance:
                print(f"    Converged with nelement={nelement}, diff={linf_abs_diff:.2e}")
                return current_result
        
        previous_result = current_result.copy()
    
    print(f"    Warning: Used finest mesh: nelement={ref_element[-1]}")
    return current_result


def run_convergence_study(lam_values, tau_values, nelement_values, target_order=6, expn_order=6, 
                         upsampling_factor=5, n_eval_points=1000,
                         methods=("qbmax", "qbx")):
    print(f"\n{'='*60}")
    print("QBMAX/QBX NEUMANN CONVERGENCE")
    print(f"{'='*60}")
    
    t = np.linspace(0, 1, n_eval_points, endpoint=False)
    eval_targets_h, _, eval_targets_normals_h, jac = starfish_parametrization(
        t, n_arms=5, amplitude=0.25
    )
    
    results = {}
    hmax_values = {}
    
    for method in methods:
        results[method] = {}
        for lam in lam_values:
            results[method][lam] = {}
            for tau in tau_values:
                results[method][lam][tau] = {}
    
    for nelement in nelement_values: 
        geometry_data = setup_geometry(nelement, expn_order, target_order, upsampling_factor)
        eval_expansion_radii_h = jac / (2 * nelement)
        
        hmax = geometry_data[-1]
        hmax_values[nelement] = hmax
        print(f"\nN = {nelement}, hmax = {hmax:.6f}")
        
        for lam in lam_values:
            print(f"  k = {lam}")

            densities = {}
            gmres_iters_by_method = {}
            for method in methods:
                print(f"  method = {method.upper()}")
                density, gmres_iters = solve_neumann_system(
                    geometry_data, lam, nelement, target_order,
                    upsampling_factor, expn_order, method)
                densities[method] = density
                gmres_iters_by_method[method] = gmres_iters
            
            for tau in tau_values:
                ref_solution = generate_reference_solution(
                    lam, tau, eval_targets_h, -2, eval_expansion_radii_h, eval_targets_normals_h
                )

                for method in methods:
                    eval_result = evaluate_solution(
                        densities[method], geometry_data, eval_targets_h,
                        eval_targets_normals_h, eval_expansion_radii_h, lam,
                        tau, expn_order, method)

                    absolute_error = np.max(np.abs(eval_result - ref_solution))
                    relative_error = absolute_error / np.max(np.abs(ref_solution))

                    results[method][lam][tau][nelement] = {
                        'eval': eval_result,
                        'ref': ref_solution,
                        # Compatibility alias retained for older post-processing.
                        'error': absolute_error,
                        'absolute_error': absolute_error,
                        'relative_error': relative_error,
                        'gmres_iters': gmres_iters_by_method[method]
                    }

                    print(
                        f"    {method.upper()}, τ = {tau}: "
                        f"abs error = {absolute_error:.2e}, "
                        f"relative error = {relative_error:.2e}")
            
            print()  
    
    return results, hmax_values


def visualize_results(data, output_path):
    """Plot the relative error per tau, the fitted order of QBMAX at tau = 1
    over the first five mesh levels, and the GMRES iteration counts."""
    import matplotlib.pyplot as plt

    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath,amssymb}")
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 6.5,
        "figure.titlesize": 12,
    })

    results = data["results_by_method"]      # results[method][k][tau][N] -> record
    params = data["parameters"]
    lams = params["lam_values"]
    nelements = params["nelement_values"]
    h_values = np.array([float(np.asarray(params["hmax_values"][n]).reshape(-1)[0]) for n in nelements])
    taus = [1.0, 0.5]
    fit_start, fit_stop = 0, 5  # fit QBMAX at tau = 1 over the first five levels

    COLORS = {1.0: "#002147", 0.5: "#B22222"}
    LINESTYLES = {"qbmax": "-", "qbx": "--"}
    MARKERS = {"qbmax": "o", "qbx": "s"}


    def relative_error(record):
        """max_i |u - u_ref| / max_i |u_ref|, as defined in the manuscript."""
        evaluation = np.asarray(record["eval"])
        reference = np.asarray(record["ref"])
        return float(np.max(np.abs(evaluation - reference)) / np.max(np.abs(reference)))


    fig, axes = plt.subplots(2, len(lams), figsize=(10, 5.5), sharex=False, sharey="row")

    for col, lam in enumerate(lams):
        ax_err, ax_gmres = axes[0, col], axes[1, col]

        for method in ("qbmax", "qbx"):
            for tau in taus:
                records = results[method][lam][tau]
                errors = [relative_error(records[n]) for n in nelements]
                ax_err.loglog(h_values, errors, color=COLORS[tau], linestyle=LINESTYLES[method],
                              marker=MARKERS[method], linewidth=1.5, markersize=4,
                              label=f"{method.upper()}, $\\tau = {tau:g}$")

        fit_errors = np.array([relative_error(results["qbmax"][lam][1.0][n]) for n in nelements])
        slope, intercept = np.polyfit(np.log10(h_values[fit_start:fit_stop]),
                                      np.log10(fit_errors[fit_start:fit_stop]), 1)
        fit_h = np.geomspace(h_values[-1], h_values[0], 100)
        ax_err.loglog(fit_h, 10 ** (slope * np.log10(fit_h) + intercept), "k--", linewidth=1.2,
                      alpha=0.8, label=rf"$\mathcal{{O}}(h^{{{slope:.1f}}})$")
        ax_err.set_xlabel(r"$h$")
        ax_err.set_ylabel(r"relative error $(\tau)$, $\ell^\infty$")
        ax_err.set_title(rf"$k = {lam}$ -- $h$-convergence")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc="best")

        for method in ("qbmax", "qbx"):
            records = results[method][lam][1.0]
            iterations = [int(records[n]["gmres_iters"]) for n in nelements]
            ax_gmres.plot(nelements, iterations, color="black" if method == "qbmax" else "#708090",
                          linestyle=LINESTYLES[method], marker=MARKERS[method], linewidth=1.5,
                          markersize=4, label=f"{method.upper()} GMRES")
        ax_gmres.set_xlabel(r"$N$")
        ax_gmres.set_ylabel("GMRES iterations")
        ax_gmres.set_title(rf"$k = {lam}$ -- GMRES vs. $N$")
        ax_gmres.grid(True, alpha=0.3)
        ax_gmres.legend(loc="best")

    fig.suptitle(rf"Interior Neumann Problem -- $p={params['expn_order']}$, "
                 rf"$q={params['target_order'] + 1}$, $\kappa={params['upsampling_factor']}$")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    print(f"Figure saved to {output_path}")
    plt.show()


def main():
    target_order = 6
    expn_order = 6
    upsampling_factor = 5
    
    lam_values = [10, 20, 40, 80]
    tau_values = [0.0, 0.5, 1.0]
    nelement_values = [
        100, 200, 300, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
    ]
    methods = ["qbmax", "qbx"]
    n_eval_points = 1000
    
    results, hmax_values = run_convergence_study(
        lam_values, tau_values, nelement_values, target_order, expn_order, 
        upsampling_factor, n_eval_points, methods
    )
    
    data_to_save = {
        'results': results['qbmax'],
        'results_by_method': results,
        'parameters': {
            'methods': methods,
            'lam_values': lam_values,
            'tau_values': tau_values,
            'nelement_values': nelement_values,
            'hmax_values': hmax_values, 
            'target_order': target_order,
            'expn_order': expn_order,
            'upsampling_factor': upsampling_factor,
            'n_eval_points': n_eval_points,
            'error_definition': (
                "error and absolute_error are max(abs(eval-ref)); "
                "relative_error is absolute_error/max(abs(ref))")
        }
    }
    
    output_path = Path(__file__).resolve().parent / "Neumann.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Results saved to {output_path}")

    print("\nVisualizing results...\n")
    visualize_results(data_to_save, output_path.with_suffix(".pgf"))
    
    
    
if __name__ == "__main__":
    main()
