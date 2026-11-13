# References

The method and its analysis:

* Z. Khalid, "An $O(L^3)$ spherical harmonic transform on $L^2$ samples",
  arXiv preprint [arXiv:2611.16441](https://arxiv.org/abs/2611.16441), 2026
  (the interpolated elimination, the exactness theorem, the
  $\Lambda$-placement and the error analysis implemented here).
* Z. Khalid, R. A. Kennedy and J. D. McEwen, "An optimal-dimensionality
  sampling scheme on the sphere with fast spherical harmonic transforms",
  *IEEE Transactions on Signal Processing*, vol. 62, no. 17, pp. 4597-4610,
  2014. The sampling scheme, the $\kappa$-placement and the linear-solve
  transform ({func}`osht.forward_lu`); MATLAB package NSHT.
* G. E. O. Giacaglia and C. A. Lundquist, "Sampling functions for geophysics",
  Smithsonian Astrophysical Observatory Special Report 344, 1972. The order-$m$
  sampling functions (their eq. 4.26), the Lagrange basis of the profile space.
* J. D. McEwen and Y. Wiaux, "A novel sampling theorem on the sphere", *IEEE
  Transactions on Signal Processing*, vol. 59, no. 12, pp. 5876-5887, 2011.
  The equiangular ring set, the periodic extension, the reflected weights and
  the Wigner step of the equiangular back-end.
* S. Trapani and J. Navaza, "Calculation of spherical harmonics and Wigner d
  functions by FFT. Applications to fast rotational matching in molecular
  replacement and implementation into AMoRe", *Acta Crystallographica A*,
  vol. 62, pp. 262-269, 2006. The recursion for the planes $d^\ell(\pi/2)$.
* U. Nafees, Z. Khalid and R. A. Kennedy, "Differentially-constrained
  reconstruction and multi-pass refinement on the sphere", Proc. SampTA,
  pp. 87-91, 2017. The multi-pass refinement and its stopping rule.
* J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
  *SIAM Review*, vol. 46, no. 3, pp. 501-517, 2004. The barycentric forms.

Related software:

* NSHT (MATLAB), the optimal-dimensionality transform of the 2014 paper.
* SSHT / `pyssht`, `s2fft`, `ducc0`, `pyshtools` and `healpy`: spherical
  harmonic transforms on redundant sampling schemes (equiangular,
  Gauss-Legendre, HEALPix), the natural choice when more than $L^2$ samples
  are available or when spin transforms are needed.
