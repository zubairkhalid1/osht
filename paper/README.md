# Experiments of the paper

This folder holds the numerical results of the accuracy and run-time
experiments reported in

> Z. Khalid, "An O(L^3) spherical harmonic transform on L^2 samples",
> arXiv preprint [arXiv:2611.16441](https://arxiv.org/abs/2611.16441), 2026,

together with what is needed to draw the figures from them or to recompute
them with `osht`. The figures themselves are in the paper and are not stored
here; the two figures of the README and of the documentation are drawn from
the same results by `docs/make_readme_figures.py`.

| File | Content | Produced by |
| --- | --- | --- |
| `accuracy_vs_L.npz` | errors of every test signal: `E_max`, `E_mean`, `SNR` with shape (placement, L, pass, signal), placements Lambda and kappa, L = 16, 32, ..., 2048, 5 signals per L | `demos/demo08_accuracy_vs_L.py` |
| `runtime.csv` | run time of one inverse transform and of one forward pass, and the single-pass E_max, for L = 16, 32, ..., 2048 | `demos/demo07_runtime_vs_L.py` |
| `plot_runtime.py` | draws the run-time figure of the paper from `runtime.csv` | |

The accuracy figure of the paper (three panels: mean of E_mean, E_max and SNR
over the signals after J = 1 and J = 2 passes, Lambda- against
kappa-placement) is drawn from `accuracy_vs_L.npz` by
`demos/demo08_accuracy_vs_L.py`, and the run-time figure from `runtime.csv`
by `plot_runtime.py`.

In the arXiv version of the paper the accuracy figure is Fig. 6 and the
run-time figure is Fig. 7.

## Protocol

Accuracy: for every band-limit and every test signal, coefficients with real
and imaginary parts uniform in [-1, 1] (seed `2026 + 1000 L + s` for signal
`s`) are synthesised on the L^2 samples by `osht.inverse` and recovered by
`osht.forward_pass` (interpolated elimination with the equiangular back-end),
followed by further passes on the sample residual with the stopping rule of
`osht.multipass`. Three passes are stored for L <= 1024 and two for L = 2048;
the figure shows J = 1 and J = 2. By the exactness of the transforms, any
difference between the two sets of coefficients is numerical error.

Run time: wall-clock time of one `osht.forward_pass` (the complete pass from
samples to coefficients on the Lambda-placement) and of one `osht.inverse`;
best of 5 repeats for L <= 256, of 3 for L = 512 and a single run for
L >= 1024. The ring placement is looked up before the clock starts.

Machine of the stored results: Windows 11 laptop, Intel Core i7-1365U
(10 cores, 12 threads), 32 GB RAM; Python 3.13.2, NumPy 2.5.4, SciPy 1.18.1.

## Results in numbers

Mean over the five signals, Lambda-placement (kappa-placement in brackets):

| L | E_max, one pass | E_max, two passes | SNR, two passes (dB) | forward pass (s) | inverse (s) |
| --- | --- | --- | --- | --- | --- |
| 16 | 5.1e-15 (5.4e-15) | 9.1e-16 (9.6e-16) | 309 (308) | 0.003 | 0.001 |
| 64 | 9.0e-14 (9.0e-14) | 3.0e-15 (3.9e-15) | 300 (299) | 0.034 | 0.012 |
| 256 | 1.3e-12 (1.3e-11) | 3.7e-14 (1.7e-13) | 284 (275) | 0.88 | 0.27 |
| 512 | 2.2e-11 (7.8e-11) | 2.8e-13 (1.3e-12) | 270 (262) | 6.3 | 2.3 |
| 1024 | 1.8e-9 (2.3e-8) | 1.5e-11 (1.5e-10) | 239 (223) | 49 | 21 |
| 2048 | 1.6e-5 (8.6e-2) | 1.6e-7 (4.1e-4) | 163 (102) | 390 | 290 |

## Reproducing

From the root of the repository, with `pip install -e ".[plot]"`:

```bash
# draw the figures from the stored results (seconds); they are written into paper/
python demos/demo08_accuracy_vs_L.py --out paper --Lmax 2048 --plot-only
python paper/plot_runtime.py

# recompute (the stored results are kept; delete or rename them to start afresh)
python demos/demo08_accuracy_vs_L.py --out paper --Lmax 2048   # about four hours, nearly all at L = 2048
python demos/demo07_runtime_vs_L.py 2048 paper/runtime         # about half an hour
python paper/plot_runtime.py
```

`demo08` saves every signal as soon as it is done and resumes from the saved
file, so a run can be stopped and restarted; `--Lmax 1024` takes about half
an hour. The accuracy figures use Times and the run-time figure uses Computer
Modern (through LaTeX when it is installed), so the fonts of a figure depend
on the machine. Run times depend on the machine; the errors are
reproducible up to rounding differences between BLAS and FFT libraries.
