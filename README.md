## Numerical Experiments and Reproducible Code for the QBMAX Paper

This project contains numerical experiments and reproducible code for the QBMAX paper.

## Dependencies
This project depends on [**pytential**](https://documen.tician.de/pytential/). 
**We strongly recommend** following the official install guide: https://documen.tician.de/pytential/misc.html

## Examples Structure
Below are the code examples included in the repo. For other numerical test results, code is available upon request.
```
QBMAX Paper Examples
├── 2D Problems
│   ├── Circle Domain
│   │   └── Close evaluations of single and double layer potentials
│   └── Starfish Domain
│       ├── Close evaluation of double layer potentials
│       ├── Jump relations of double layer potentials
│       ├── Oscillatory Helmholtz
│       └── Interior Neumann BVP
├── 3D Problems
│   └── Cruller
│       └── Jump relations of double layer potentials
└── Cost Analysis
    └── Weighted operation counts of QBX and QBMAX expansions (SymPy)
```

The interior Neumann example runs both QBX and QBMAX, stores the relative error
of each evaluation, and reproduces the convergence figure of the paper from the
saved `Neumann.pkl`. The cost-analysis scripts regenerate `flop_count_2d.pkl`
and `flop_count_3d.pkl` (symbolic differentiation up to order 8, which takes a
while) and `plot_flop_count.py` draws the figure from them.