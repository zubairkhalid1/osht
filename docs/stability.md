# Accuracy and stability

All transforms of the package are exact in exact arithmetic. In floating
point the only non-exact step of the forward transform is the interpolation
to the small rings, whose worst-case amplification for order $m$ is

$$
\Lambda_m = \max_{k < |m|} \sum_{s \ge |m|} |\Lambda^m_s(\theta_k)| = \|P_m^{(<m)} P_m^{-1}\|_\infty .
$$

The interpolation error of order $m$ on a small ring is a pure $e^{\pm im\phi}$,
aliases onto one DFT bin of that ring, is amplified by about $\Lambda$ at the
next order and injected again; the single-pass error therefore grows along the
elimination chain towards $m = 0$ (the *oracle sweep* of
{func}`osht.forward_pass`, which cleans with the exact profiles, keeps every
order at rounding level and isolates this propagation). The levers are a small
$\Lambda_m$ (the ring placement), exact local steps (FFTs, log-domain sampling
functions, the rescaled Legendre recurrence) and the residual refinement.
{func}`osht.amplification` returns $\Lambda_m$, $\Lambda_m^{GL}$ and
$\kappa(P_m)$ for any placement, and `return_info=True` returns the constants
of the sweep.

## Scalar transforms

Measured in this package (NumPy, double precision, random complex signals
with real and imaginary parts uniform in $[-1, 1]$, $\Lambda$-placement,
equiangular back-end, mean over five signals of
$E_{\max} = \max |f_\ell^m - \tilde f_\ell^m|$; the experiment of
`demos/demo08_accuracy_vs_L.py`, see {doc}`demos`):

| $L$ | $\max_m \Lambda_m$ | one pass | two passes | three passes | linear solves (2014) |
| --- | --- | --- | --- | --- | --- |
| 64 | 46 | 9.0e-14 | 3.0e-15 | 3.7e-15 | 1e-14 |
| 128 | 100 | 3.0e-13 | 1.2e-14 | 1.3e-14 | 2e-14 |
| 256 | 259 | 1.3e-12 | 3.7e-14 | 3.2e-14 | 8e-14 |
| 512 | 348 | 2.2e-11 | 2.8e-13 | 2.2e-13 | 4e-13 |
| 1024 | 1941 | 1.8e-9 | 1.5e-11 | 1.4e-11 | |
| 2048 | 2843 | 1.6e-5 | 1.6e-7 | | |

```{image} _static/accuracy.png
:alt: Maximum coefficient error versus band-limit after one and two passes, for the Lambda- and the kappa-placement
:width: 80%
:align: center
```

These agree with the MATLAB package (one pass 2.0e-12 and two passes
3.8e-14 at $L = 256$; 2.4e-11 and 1.9e-13 at $L = 512$; 2.6e-9, 1.3e-11 and
1.5e-11 after one, two and three passes at $L = 1024$ with ten signals). A
third pass gains nothing over the second. In terms of the signal-to-noise
ratio, two passes give 284 dB at $L = 256$, 239 dB at $L = 1024$ and 163 dB
at $L = 2048$.

With the $\kappa$-placement the errors are the same up to $L = 128$ and
larger beyond: after one pass by a factor of 10 at $L = 256$, 4 at $L = 512$
and 13 at $L = 1024$, and at $L = 2048$ the single pass fails (0.09, against
1.6e-5 for the $\Lambda$-placement) and two passes reach only 4e-4. At
$L = 1536$ the $\Lambda$-placement gives 2e-6 after one pass and 1.6e-8
after two.

The GL back-end is as accurate as the equiangular one (the latter is two to
three times better at $L \le 128$). The real-signal path has the same accuracy
as the complex path. Noise on the samples is amplified by about $\Lambda$: with
a relative sample noise of $10^{-8}$ the single-pass error is 4e-6 at
$L = 256$ and 2.5e-5 at $L = 512$, which is the price of having no redundancy
in the samples (the condition number of the collocation map grows like
$L^{1.5}$ against a small and flat value for redundant schemes).

{func}`osht.forward_pass` emits a `RuntimeWarning` when the amplification of
the sweep exceeds $10^4$, and {func}`osht.forward` (through
{func}`osht.multipass`) when the relative sample residual before the last pass
exceeds $10^{-9}$ (healthy residuals are at $10^{-12}$ or below, and the
residual on exactly $L^2$ samples is not affected by noise in the samples);
both signal a placement or a band-limit at which the transform is not
accurate.

## The alternating placement

The analytic alternating placement of the 2014 paper is only usable at small
$L$: its amplification reaches $10^9$ at $L = 128$ and both the interpolated
elimination and the linear solves fail there. Use the stored $\Lambda$- or
$\kappa$-placements, or run a greedy search ({doc}`placements`).

## Spin signals are not supported

The package transforms scalar signals only. Spin-$s$ transforms on the
$L^2$ samples are exact in exact arithmetic (the equiangular back-end handles
them unchanged), but the sweep is numerically unstable beyond small band-limits
with the available placements: the profile of order $m$ of a spin-$s$ signal
vanishes at one of the poles only like the scalar order $|m| - |s|$, so the
pole-adjacent small rings carry order-$m$ content that is not small and the
interpolation to them is amplified by about $(2L/\pi)^{2|s|}$ relative to the
scalar case. The linear-solve transform of the 2014 paper fails identically,
because this is the conditioning of the sampling set for spin signals, not a
property of the algorithm (measured: $|s| = 1$ gives 2e-10 at $L = 16$ and
4e-4 at $L = 32$; $|s| = 2$ gives 4e-10 at $L = 8$ and fails at $L = 16$), and a
greedy placement with a spin objective does not help. For spin transforms use a
redundant sampling theorem, for example the equiangular scheme of McEwen and
Wiaux in `pyssht`, `s2fft` or `ducc0`.

## Where the floor comes from

Two passes of the refinement reach a floor that further passes do not lower
although the sample residual is at rounding level: the floor is the rounding
of the samples amplified by the inverse of the collocation map (about $10^4$
for the $\Lambda$-placement at $L = 1536$). The floor rises from about
1e-11 at $L = 1024$ to 1.6e-8 at $L = 1536$ and 1.6e-7 at $L = 2048$.
