# Demos and the experiments of the paper

The `demos/` folder of the repository contains runnable scripts. Each one
takes the band-limit or its options on the command line, prints what it
measures and, where it plots, writes its figures to the current directory.

| Script | Shows |
| --- | --- |
| `demo01_quickstart.py [L]` | a spectral-spatial-spectral round trip with both back-ends, one and two passes, and the linear-solve reference |
| `demo02_real_signal_earth.py [L]` | the Earth topography with the real-signal fast path; plots of the signal and of the error spectrum |
| `demo03_placements_and_amplification.py [L]` | the three ring placements, their amplification constants and their accuracy; the greedy searches against the stored placements |
| `demo04_multipass_and_oracle.py [L]` | the multi-pass refinement with its stopping rule, and the oracle sweep that isolates the error propagation |
| `demo05_sampling_functions_and_new_L.py` | the order-$m$ sampling functions, and a band-limit without a shipped placement (search once, cached afterwards) |
| `demo06_batch_and_timing.py [Lmax]` | batches of signals and timings of the inverse, both back-ends and the linear solves |
| `demo07_runtime_vs_L.py [Lmax] [prefix]` | run time of one forward pass and one inverse for $L = 16, 32, \dots, L_{\max}$ (default 2048), saved to CSV and plotted |
| `demo08_accuracy_vs_L.py [options]` | $E_{\rm mean}$, $E_{\max}$ and SNR versus $L$ after $J = 1, 2$ passes for the $\Lambda$- and the $\kappa$-placement; resumable |

## The experiments of the paper

Demos 7 and 8 are the run-time and the accuracy experiment of the paper
([arXiv:2611.16441](https://arxiv.org/abs/2611.16441)). The
folder `paper/` of the repository stores their numerical results
(`runtime.csv`, `accuracy_vs_L.npz`) and a description of the protocol; the
figures themselves are in the paper and are drawn from the stored results by
`demo08` and by `paper/plot_runtime.py`. From the root of the repository:

```bash
# draw the figures from the stored results (seconds)
python demos/demo08_accuracy_vs_L.py --out paper --Lmax 2048 --plot-only
python paper/plot_runtime.py

# recompute
python demos/demo08_accuracy_vs_L.py --out paper --Lmax 2048   # about four hours, nearly all at L = 2048
python demos/demo07_runtime_vs_L.py 2048 paper/runtime         # about half an hour
```

`demo08` saves every signal as soon as it is done and resumes from the saved
file, so a run can be stopped and restarted, and a later run with a larger
`--Lmax` only computes the new band-limits.

## Run time

One forward pass (the complete pass from samples to coefficients: sweep and
equiangular back-end) and one inverse transform on the $\Lambda$-placement,
pure NumPy, on a laptop (Intel Core i7-1365U, 32 GB RAM, Python 3.13,
NumPy 2.5, SciPy 1.18):

| $L$ | forward pass (s) | inverse (s) |
| --- | --- | --- |
| 64 | 0.034 | 0.012 |
| 128 | 0.14 | 0.062 |
| 256 | 0.88 | 0.27 |
| 512 | 6.3 | 2.3 |
| 1024 | 49 | 21 |
| 2048 | 390 | 290 |

```{image} _static/runtime.png
:alt: Run time of one forward pass and one inverse transform versus band-limit
:width: 80%
:align: center
```

The forward time grows by a factor of 7.2, 7.8 and 8.0 per doubling from
$L = 256$, the $O(L^3)$ scaling. The default transform of {func}`osht.forward`
runs two passes and one synthesis of the residual, about two forward passes
and one inverse.
