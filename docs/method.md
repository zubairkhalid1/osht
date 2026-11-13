# The method

This page summarises the mathematics behind the package. The full statement
with proofs is in the paper (see {doc}`references`).

## Sampling scheme

A signal band-limited at $L$ has $L^2$ degrees of freedom. The scheme takes
$L$ iso-latitude rings, ring $k$ at co-latitude $\theta_k$ with $2k+1$
equally spaced samples, hence $\sum_{k=0}^{L-1} (2k+1) = L^2$ samples. The
ring co-latitudes are the equiangular set
$\Theta = \{ (2t+1)\pi/(2L-1) : t = 0, \dots, L-1 \}$ of McEwen and Wiaux, and
the *placement* assigns them to the rings (see {doc}`placements`).

## Order profiles and the elimination

Write $G_m(\theta) = \int_0^{2\pi} f(\theta, \phi)\, e^{-im\phi}\, d\phi$ for the
order-$m$ profile. On ring $k$ the $(2k+1)$-point DFT of the samples resolves
the orders $|m| \le k$ without aliasing once the orders $|m| > k$ have been
removed. The forward transform therefore sweeps $m = L-1, \dots, 0$:

1. **Resolve.** $G_{\pm m}(\theta_k)$ on the rings $k \ge |m|$ from the DFT bins.
2. **Interpolate.** $G_{\pm m}(\theta_k)$ on the rings $k < |m|$ from the
   resolving rings by the order-$m$ sampling functions (below).
3. **Clean.** Subtract $\frac{1}{2\pi}\big(G_m(\theta_k) e^{im\phi} + G_{-m}(\theta_k) e^{-im\phi}\big)$
   from every ring $k < |m|$.

After the sweep the profiles of every order are known on all $L$ rings and a
back-end turns them into coefficients. The 2014 transform solves instead, for
every order, the $(L-m)\times(L-m)$ collocation system $P_m f^m = g_m$
({func}`osht.forward_lu`); the interpolation realises $P_m^{(<m)} P_m^{-1}$ in
closed form, which removes the $O(L^4)$ solves and the storage of the factors.

## Sampling functions

The order-$m$ profile of a band-limited signal lies in
$V_m = \sin^{|m|}\theta \times \{\text{polynomials of degree} \le L-1-|m| \text{ in } \cos\theta\}$,
a space of dimension $L - |m|$, the number of resolving rings. Its Lagrange
basis on the resolving co-latitudes is the set of sampling functions of
Giacaglia and Lundquist (1972),

$$
\Lambda^m_s(\theta) = \Big(\frac{\sin\theta}{\sin\theta_s}\Big)^{|m|}
\prod_{r \ge |m|, r \ne s} \frac{\cos\theta - \cos\theta_r}{\cos\theta_s - \cos\theta_r},
\qquad
G_m(\theta) = \sum_{s \ge |m|} G_m(\theta_s)\, \Lambda^m_s(\theta),
$$

so step 2 is exact. The package evaluates them in the first barycentric form
in the log domain ({func}`osht.lambda_matrix`): every entry is accurate to a
relative error of $O(L\,\epsilon_{\rm mach})$ even when the individual factors
overflow or underflow, and the barycentric weights of the nested node sets
$\{\theta_m, \dots, \theta_{L-1}\}$ are updated in $O(L-m)$ as $m$ decreases.

## Back-ends

**Equiangular** (`backend="eq"`, default). The ring set is the equiangular
set, so the profiles on all $L$ rings are exactly the input of the forward
algorithm of McEwen and Wiaux (2011): periodic extension in $\theta$ with the
factor $(-1)^m$, FFT, convolution with the reflected weights
$w(m') = \int_0^\pi \sin\theta\, e^{im'\theta} d\theta$, and the Wigner step

$$
f_\ell^m = i^{m} \sqrt{\tfrac{2\ell+1}{4\pi}}
\sum_{m'} \Delta^\ell_{m'm}\, \Delta^\ell_{m'0}\, G_{mm'},
\qquad \Delta^\ell = d^\ell(\pi/2),
$$

with the planes $\Delta^\ell$ from the Trapani-Navaza recursion
({func}`osht.trapani_plane`).

**Gauss-Legendre** (`backend="gl"`). The profiles are interpolated by the same
sampling functions to the $L$ Gauss-Legendre nodes and
$f_\ell^m = \sum_i w_i\, G_m(\theta_i)\, \tilde P_\ell^m(\theta_i)$, exact because
$G_m \tilde P_\ell^m$ has degree at most $2L-2$ in $\cos\theta$. This back-end
works for any set of distinct ring co-latitudes.

## Inverse transform

$G_m(\theta_k) = 2\pi \sum_{\ell \ge |m|} f_\ell^m\, \tilde P_\ell^m(\theta_k)$
on every ring, and ring $k$ is synthesised by an inverse FFT of length $2k+1$
of the orders folded modulo $2k+1$; the aliasing of the orders $|m| > k$ onto
the $2k+1$ samples is exact, not an approximation ({func}`osht.inverse`).

## Multi-pass refinement

With $\tilde f$ the estimate, the residual $r = f - \mathrm{inverse}(\tilde f)$
is transformed and added, and the iteration stops when the maximum of the
residual stops decreasing (Nafees et al., 2017). Two passes reach a floor set
by the sample rounding amplified by the inverse collocation map
({func}`osht.multipass`).

## Cost

Per pass: $O(L^2 \log L)$ for the ring FFTs, $L^3/6$ for the interpolation,
$L^3/3$ for the cleaning in the DFT domain, and $O(L^3)$ for the back-end;
no linear solve and no precomputation beyond the ring co-latitudes. The
cleaning is applied to one DFT bin per ring instead of to the samples (the two
forms are identical in exact arithmetic), which keeps the sweep at $O(L^3)$.
