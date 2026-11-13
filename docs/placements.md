# Ring placements

The ring co-latitudes are the equiangular set
$\Theta = \{(2t+1)\pi/(2L-1)\}$; a placement assigns them to the rings, that
is, decides which co-latitude gets the ring with $2k+1$ samples. Every
placement gives an exact transform; the placement decides the stability.
Three are available through {func}`osht.ordered_theta` and every function
that takes `thetas`:

`"lambda"` (default)
: Greedy minimisation of the interpolation amplification $\Lambda_m$: the
  largest ring is placed nearest the equator and, for $m = L-2, \dots, 0$, the
  unused co-latitude that minimises $\Lambda_m$ (the amplification of the
  interpolation to the rings that will be cleaned at step $m$) becomes
  $\theta_m$. Stored for $L = 1, \dots, 128$ and $L = 256, 512, 1024, 1536, 2048$;
  {func}`osht.search_lambda_placement` runs the search ($O(L^4)$: seconds at
  $L = 128$, two minutes at $L = 512$, half an hour at $L = 1024$, about
  40 hours on a laptop at $L = 2048$) and saves
  the result in the user cache; `osht.ordered_theta(L, search=True)` does the
  same and is the usual way to get a placement for a new band-limit.

`"kappa"`
: Greedy minimisation of the condition number $\kappa(P_m)$ of the per-order
  collocation matrix, the placement of the 2014 paper and of the NSHT package.
  Stored for $L = 1, \dots, 721$ and $L = 1024, 1536, 2048$;
  {func}`osht.search_kappa_placement` runs the search in $O(L^4)$ through a
  rank-one update of the extreme singular values (the reference search with a
  full SVD per candidate, `exact=True`, is $O(L^5)$ and gives the same result),
  also saved in the user cache.

`"alternating"`
: The analytic ordering of the 2014 paper ({func}`osht.indexed_theta`): the
  largest ring nearest the equator, then alternating between the hemispheres
  outwards, the smallest ring at the south pole. Unstable beyond small $L$.

Both greedy placements keep the largest rings nearest the equator, alternating
between hemispheres as the alternating ordering does, and depart from it mainly
for rings of intermediate size, which they move to mid-latitudes.

| $L$ | $\max_m \Lambda_m$ ($\Lambda$ / $\kappa$) | $\max_m \Lambda_m^{GL}$ ($\Lambda$ / $\kappa$) | $\max_m \kappa(P_m)$ ($\Lambda$ / $\kappa$) |
| --- | --- | --- | --- |
| 64 | 46 / 46 | 38 / 42 | 11.2 / 10.3 |
| 128 | 100 / 171 | 84 / 190 | 21.4 / 30.7 |
| 256 | 259 / 609 | 185 / 525 | 49 / 65 |
| 512 | 348 / 2432 | 346 / 2520 | 61 / 209 |
| 1024 | 1941 / 3752 | | 165 / 213 |
| 1536 | 3228 / 5408 | | |
| 2048 | 2843 / 14200 | | |

The $\Lambda$-placement also lowers $\kappa(P_m)$, so it improves the
linear-solve transform by the same factor as the interpolated elimination.

## The placement cache

A placement computed by either search for a band-limit that is not shipped
is written to the user cache, `~/.cache/osht` on Linux and macOS,
`%LOCALAPPDATA%\osht` on Windows, or `OSHT_CACHE_DIR` if that environment
variable is set ({func}`osht.cache_dir`), as a small `.npz` file holding the
index permutation of the equiangular set and the objective of the search.
{func}`osht.ordered_theta` looks there after the shipped placements, so every
later call of the package at that band-limit, in any session, uses it
without recomputation. {func}`osht.cached_placements` lists the cached
band-limits and {func}`osht.save_placement` stores a placement obtained
elsewhere, for instance the `theta_lambda_L.mat` files of the MATLAB package
(load them with `scipy.io.loadmat` and pass `TT_updated`).

## Other ring sets

{func}`osht.forward_pass` with `backend="gl"`, {func}`osht.forward_lu` and
{func}`osht.inverse` accept any set of $L$ distinct co-latitudes in $[0, \pi]$,
for example the Gauss-Legendre nodes in alternating order. The equiangular
back-end needs the equiangular set. The stability of other ring sets is not
studied; the amplification constants of {func}`osht.amplification` tell.
