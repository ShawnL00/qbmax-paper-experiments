"""
Jump relations verification for the double layer potential Dₖσ 
on the cruller domain, where the density function is defined as:
σ(x) := sin(3πx₁) cos(5πx₂) sin(10x₃) + 2 for x = (x₁, x₂, x₃).
"""

import numpy as np
import pickle
import os
from pathlib import Path

# Parameters
nlevel = [4, 6, 8, 10, 12]
lams = [10, 20, 40, 80]
expn_order = 4
target_order = 4
upsampling_factor = 6
    
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

results_file = "D_jump_convergence.pkl"
file_exists = os.path.exists(results_file)

import matplotlib.pyplot as plt

def visualize_results(data):
    from cycler import cycler
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
    
    qbx_results = data['qbx']
    qbmax_results = data['qbmax']
    nlevel = data['nlevel']
    lams = data['lams']
    expn_order = data['expn_order']
    target_order = data['target_order']
    upsampling_factor = data['upsampling_factor']
    
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.15))
    
    qbx_marker = '<'
    qbmax_marker = 'o'
    
    for idx, lam in enumerate(lams):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][idx % 5]
        
        qbx_h = qbx_results[lam]["h"][1:]
        qbx_err = [np.max(err) for err in qbx_results[lam]['abs_error'][1:]]
        ax.plot(np.log10(qbx_h), np.log10(qbx_err), marker=qbx_marker, 
                label=fr'QBX, $k = {lam}$', color=color, linestyle='--')
        
        qbmax_h = qbmax_results[lam]["h"][1:]
        qbmax_err = [np.max(err) for err in qbmax_results[lam]['abs_error'][1:]]
        ax.plot(np.log10(qbmax_h), np.log10(qbmax_err), marker=qbmax_marker, 
                label=fr'QBMAX, $k = {lam}$', color=color)
    
    ax.set_ylabel(r"$\log_{10}(\mathrm{err})$", fontsize=10)
    ax.set_xlabel(r"$\log_{10}(h)$", fontsize=10)
    ax.set_title(rf"Jump Relations of $\mathcal{{D}}_k\sigma$ "
                 rf"($p={expn_order}$, $q={target_order + 1}$, "
                 rf"$\kappa={upsampling_factor}$)")
    
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        bbox_to_anchor=(1.05, 0.5),  
        loc='center left',           
        ncol=1,                      
        fontsize=9,
        frameon=True,
        handlelength=2,
    )
    
    fig.subplots_adjust(top=0.85, right=0.7)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


if file_exists:
    print("Loading and plotting data...")
    
    with open(results_file, "rb") as f:
        data = pickle.load(f)
        visualize_results(data)

else:
    print("Results file not found. Running computation...")
    import pyopencl as cl
    from meshmode.array_context import PyOpenCLArrayContext

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)
    
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
    
    def setup_geometry(level, expn_order, target_order, upsampling_factor=1):
        """Set up mesh, discretization, and QBX source for cruller geometry."""
        from arraycontext import flatten
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import InterpolatoryQuadratureGroupFactory
        from meshmode.mesh import TensorProductElementGroup
        import meshmode.mesh.generation as mgen
        from pytential import GeometryCollection, bind, sym
        from pytential.qbx import QBXLayerPotentialSource
        
        mesh = mgen.generate_cruller(
            r_major=4, r_minor=2,
            n_major=20 * level, n_minor=10 * level,
            order=target_order, group_cls=TensorProductElementGroup
        )

        pre_density_discr = Discretization(
            actx, mesh, InterpolatoryQuadratureGroupFactory(target_order)
        )
        
        qbx = QBXLayerPotentialSource(
            pre_density_discr, upsampling_factor * target_order, 
            expn_order, fmm_order=False
        )
        
        places = GeometryCollection({"qbx": qbx}, auto_where=("qbx"))
        target_discr = places.get_discretization("qbx", sym.QBX_SOURCE_STAGE1)
        source_discr = places.get_discretization("qbx", sym.QBX_SOURCE_QUAD_STAGE2)
        
        targets = target_discr.nodes()
        sources = source_discr.nodes()
        ambient_dim = mesh.ambient_dim
        
        dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_QUAD_STAGE2)
        normals = bind(places, sym.normal(qbx.ambient_dim, dofdesc=dofdesc))(actx)
        normals = normals.as_vector(object)
        normals_h = actx.to_numpy(flatten(normals, actx)).reshape(ambient_dim, -1)
        
        target_dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_STAGE1)
        normals_target = bind(places, sym.normal(qbx.ambient_dim, 
                                               dofdesc=target_dofdesc))(actx)
        normals_target = normals_target.as_vector(object)
        normals_target_h = actx.to_numpy(flatten(normals_target, actx)).reshape(ambient_dim, -1)
        
        expansion_radii = bind(places, sym.expansion_radii(ambient_dim))(actx)
        centers_in = bind(places, sym.expansion_centers(qbx.ambient_dim, -1))(actx)
        centers_out = bind(places, sym.expansion_centers(qbx.ambient_dim, +1))(actx)
        weights_nodes = bind(places, sym.weights_and_area_elements(
            ambient_dim=ambient_dim, dim=ambient_dim-1, dofdesc=dofdesc))(actx)
        
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

    def make_cruller_parametrization():
        import sympy as sp
        u, v = sp.symbols("u v")
        R, r = sp.symbols("R r")
        n_major, n_minor = sp.symbols("n_major n_minor", positive=True)

        H = 1 + 0.25 * sp.cos(10 * sp.pi * u + 6 * sp.pi * v)
        x = sp.cos(2 * sp.pi * u) * (R + r * H * sp.cos(2 * sp.pi * v))
        y = sp.sin(2 * sp.pi * u) * (R + r * H * sp.cos(2 * sp.pi * v))
        z = r * H * sp.sin(2 * sp.pi * v)

        su = sp.Matrix([x.diff(u), y.diff(u), z.diff(u)])
        sv = sp.Matrix([x.diff(v), y.diff(v), z.diff(v)])
        cprod = su.cross(sv)
        normals = cprod / sp.sqrt(cprod.dot(cprod))

        su_scaled = su / n_major
        sv_scaled = sv / n_minor

        E = su_scaled.dot(su_scaled)
        F = su_scaled.dot(sv_scaled)
        G = sv_scaled.dot(sv_scaled)

        trace = E + G
        det = E * G - F**2
        eig_max = 0.5 * (trace + sp.sqrt(trace**2 - 4 * det))
        radius_expr = 0.25 * sp.sqrt(eig_max)

        pos_func = sp.lambdify((u, v, R, r), [x, y, z], "numpy")
        normal_func = sp.lambdify((u, v, R, r), normals, "numpy")
        radius_func = sp.lambdify((u, v, R, r, n_major, n_minor), radius_expr, "numpy")

        def cruller_parametrization(u_val, v_val, R_val, r_val, nmaj_val, nmin_val):
            pos = np.array(pos_func(u_val, v_val, R_val, r_val))
            normal = np.array(normal_func(u_val, v_val, R_val, r_val))
            radius = float(radius_func(u_val, v_val, R_val, r_val, nmaj_val, nmin_val))
            return pos, normal, radius

        return cruller_parametrization

    def generate_evaluation_targets(n_u=10, n_v=10, target_level=2):
        cruller_func = make_cruller_parametrization()
        u_vals = np.linspace(0, 1, n_u, endpoint=False)
        v_vals = np.linspace(0, 1, n_v, endpoint=False)
        
        eval_targets_h = np.zeros((3, n_u * n_v))
        eval_targets_normals_h = np.zeros((3, n_u * n_v))
        eval_targets_radii_h = np.zeros(n_u * n_v)
        
        idx = 0
        for j, v in enumerate(v_vals):
            for i, u in enumerate(u_vals):
                pos, normal, radius = cruller_func(u, v, 4, 2, 20*target_level, 10*target_level)
                eval_targets_h[:, idx] = pos
                eval_targets_normals_h[:, idx] = np.array(normal).flatten()
                eval_targets_radii_h[idx] = radius
                idx += 1
        
        return eval_targets_h, eval_targets_normals_h, eval_targets_radii_h

    def sigma_fun(x):
        return np.sin(3*np.pi*x[0]) * np.cos(5*np.pi*x[1]) * np.sin(10*x[2]) + 2

    from sumpy.kernel import YukawaKernel, DirectionalSourceDerivative
    knl = YukawaKernel(3)
    knl = DirectionalSourceDerivative(knl, dir_vec_name="dsource_vec")
    asym_knl = asym_yukawa(3)
    
    from sumpy.expansion.local import (
        AsymptoticDividingLineTaylorExpansion,
        LineTaylorLocalExpansion,
    )
    from sumpy.qbx import LayerPotential
    
    asymexpn = AsymptoticDividingLineTaylorExpansion(knl, asym_knl, expn_order)
    qbmax_lpot = LayerPotential(actx.context, expansion=asymexpn, 
                               source_kernels=(knl,), target_kernels=(knl,))

    expn = LineTaylorLocalExpansion(knl, expn_order)
    qbx_lpot = LayerPotential(actx.context, expansion=expn, 
                             source_kernels=(knl,), target_kernels=(knl,))

    # Generate evaluation targets
    nu = 100
    nv = 100
    eval_targets_h, eval_targets_normals_h, base_level_expansion_radii = \
        generate_evaluation_targets(nu, nv, 1)
    eval_targets = actx.from_numpy(eval_targets_h)

    qbx_results = {}
    qbmax_results = {}

    for lam in lams:
        print(f"Running k = {lam}")
        
        qbmax_results[lam] = {
            'inner': [], 'outer': [], 'sigma_true': [],
            'abs_error': [], 'r': [], 'h': []
        }
        qbx_results[lam] = {
            'inner': [], 'outer': [], 'sigma_true': [],
            'abs_error': [], 'r': [], 'h': []
        }
        
        for level in nlevel:
            print(f"  Level {level}")
            eval_targets_radii_h = base_level_expansion_radii / level
            
            eval_centers_in = actx.from_numpy(
                eval_targets_h + (-1) * eval_targets_radii_h * eval_targets_normals_h
            )
            eval_centers_out = actx.from_numpy(
                eval_targets_h + (+1) * eval_targets_radii_h * eval_targets_normals_h
            )

            (sources_h, targets_h, centers_in_h, centers_out_h, weights_nodes_h, 
             expansion_radii_h, normals_h, normals_target_h, hmax_h) = setup_geometry(
                level, expn_order, target_order, upsampling_factor)

            sources = actx.from_numpy(sources_h)
            normal = actx.from_numpy(normals_h)
            
            sigma = actx.from_numpy(sigma_fun(sources_h) * weights_nodes_h)
            strengths = (sigma,)
            extra_kwargs = {"dsource_vec": normal, "lam": lam}
            
            # QBMAX
            _, (qbmax_result_lpot1,) = qbmax_lpot(
                actx.queue, targets=eval_targets, sources=sources, 
                centers=eval_centers_in, expansion_radii=eval_targets_radii_h, 
                strengths=strengths, **extra_kwargs
            )
            _, (qbmax_result_lpot2,) = qbmax_lpot(
                actx.queue, targets=eval_targets, sources=sources, 
                centers=eval_centers_out, expansion_radii=eval_targets_radii_h, 
                strengths=strengths, **extra_kwargs
            )
            
            qbmax_result_lpot1 = actx.to_numpy(qbmax_result_lpot1)
            qbmax_result_lpot2 = actx.to_numpy(qbmax_result_lpot2)
            qbmax_error = np.abs(qbmax_result_lpot1 - qbmax_result_lpot2 + sigma_fun(eval_targets_h))
            
            qbmax_results[lam]['inner'].append(qbmax_result_lpot1)
            qbmax_results[lam]['outer'].append(qbmax_result_lpot2)
            qbmax_results[lam]['sigma_true'].append(sigma_fun(eval_targets_h))
            qbmax_results[lam]['abs_error'].append(qbmax_error)
            qbmax_results[lam]['r'].append(eval_targets_radii_h)
            qbmax_results[lam]['h'].append(hmax_h)

            # QBX
            _, (qbx_result_lpot1,) = qbx_lpot(
                actx.queue, targets=eval_targets, sources=sources, 
                centers=eval_centers_in, expansion_radii=eval_targets_radii_h, 
                strengths=strengths, **extra_kwargs
            )
            _, (qbx_result_lpot2,) = qbx_lpot(
                actx.queue, targets=eval_targets, sources=sources, 
                centers=eval_centers_out, expansion_radii=eval_targets_radii_h, 
                strengths=strengths, **extra_kwargs
            )
            
            qbx_result_lpot1 = actx.to_numpy(qbx_result_lpot1)
            qbx_result_lpot2 = actx.to_numpy(qbx_result_lpot2)
            qbx_error = np.abs(qbx_result_lpot1 - qbx_result_lpot2 + sigma_fun(eval_targets_h))
            
            qbx_results[lam]['inner'].append(qbx_result_lpot1)
            qbx_results[lam]['outer'].append(qbx_result_lpot2)
            qbx_results[lam]['sigma_true'].append(sigma_fun(eval_targets_h))
            qbx_results[lam]['abs_error'].append(qbx_error)
            qbx_results[lam]['r'].append(eval_targets_radii_h)
            qbx_results[lam]['h'].append(hmax_h)

    data = {
        "qbmax": qbmax_results,
        "qbx": qbx_results,
        'expn_order': expn_order,
        'target_order': target_order,
        'upsampling_factor': upsampling_factor,
        'nlevel': nlevel,
        'lams': lams,
        'targets': eval_targets_h,
        'targets_normal': eval_targets_normals_h,
    }

    with open(results_file, "wb") as f:
        pickle.dump(data, f)
    
    print("\nComputation completed and results saved! Visualizing results...\n")
    visualize_results(data)