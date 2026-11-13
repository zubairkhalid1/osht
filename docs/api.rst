API reference
=============

Transforms
----------

.. autosummary::
   :toctree: generated

   osht.forward
   osht.inverse
   osht.forward_pass
   osht.multipass
   osht.forward_lu
   osht.backend_eq

Sampling scheme and placements
------------------------------

.. autosummary::
   :toctree: generated

   osht.Sampling
   osht.sampling_points
   osht.ordered_theta
   osht.indexed_theta
   osht.equiangular_set
   osht.is_equiangular
   osht.theta_index
   osht.ring_index
   osht.ring_slice
   osht.stored_placements
   osht.search_lambda_placement
   osht.search_kappa_placement
   osht.cached_placements
   osht.cache_dir
   osht.save_placement

Harmonics and sampling functions
--------------------------------

.. autosummary::
   :toctree: generated

   osht.elm2ind
   osht.ind2elm
   osht.legmat
   osht.sharmonic
   osht.gauss_legendre
   osht.sampling_functions
   osht.lambda_matrix
   osht.bary_weights
   osht.trapani_plane
   osht.reflected_weights

Test signals, errors and amplification
--------------------------------------

.. autosummary::
   :toctree: generated

   osht.random_flm
   osht.errors
   osht.Errors
   osht.amplification
   osht.data.earth_topography

Plotting
--------

.. autosummary::
   :toctree: generated

   osht.plotting.plot_sampling
   osht.plotting.plot_sphere
   osht.plotting.plot_spectral

Module constants
----------------

``osht.transforms.AMPLIFICATION_WARNING`` (default ``1e4``) and
``osht.transforms.RESIDUAL_WARNING`` (default ``1e-9``) are the thresholds of
the stability warnings; set them to ``numpy.inf`` to silence the warnings.
