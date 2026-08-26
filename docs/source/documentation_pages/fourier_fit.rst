Fourier dihedral fit
====================

Amber stores a torsion as a short cosine series

.. math::

   V(\phi)=\sum_{n=1}^{n_{\mathrm{prim}}}
   K_n\bigl(1+\cos(n\phi-\gamma_n)\bigr).

Default ``nprim`` is 3 (periods 1, 2, 3) with phase 0. GenDihedFit matches
**profile shape**: the target is the mean-centered residual
``y = (E_HL - E_MM) - mean(...)``, and the design-matrix columns are the
mean-centered ``1 + cos(n phi)`` terms. Force constants enter linearly
under fixed geometry.

The explosion is identifiability, not a 7000 kcal barrier
---------------------------------------------------------

On a complete uniform 360 deg grid those columns are essentially
orthogonal and well-behaved. They become nearly linearly dependent when:

* the wavefront is gappy, clustered around one well, or failed at many
  nodes
* the HL-MM residual is **not** a smooth function of that one angle
  (steric crash, sulfate 1-4s, coupled rotors, HL noise)

Then :math:`\min \|A K - y\|^2` has a long valley: a few-kcal series and a
cancelling ``K = (4000, -3900, 200)`` series can fit the **samples**
equally well. The second one is what used to appear as PK of thousands
in a ``frcmod``.

A box :math:`|K_n|\le 25` kcal/mol is a constraint on **coefficients**,
not on **energy**. After unbounded least squares, clipping is not the
constrained LS solution. Three terms at the cap can still reconstruct a
barrier up to :math:`2\sum |K_n|` (150 kcal). That box remains only as an
Amber-safety valve so a wild PK cannot be written to a ``frcmod``.

What ffpopt does instead
------------------------

1. **Minimum-norm Fourier series (truncated SVD + Tikhonov).**
   Solve :math:`\min \|AK-y\|^2 + \lambda\|K\|^2`, dropping SVD modes with
   :math:`\sigma_i / \sigma_{\max}` below ``FFPOPT_DIHED_SVD_REL``
   (default ``1e-4``). Among cosine series that match the residual, this
   is the unique small-K one: it refuses cancelling harmonics unless the
   data need them. ``FFPOPT_DIHED_RIDGE_LAMBDA`` starts at 0 (truncated
   SVD only) and is increased if the barrier constraint below is
   violated.

2. **Energy-domain barrier on** :math:`V(\phi)`, **not on PK.**
   The reconstructed shape peak-to-peak at the sample angles must stay
   within ``FFPOPT_DIHED_BARRIER_ALPHA`` times the data peak-to-peak
   (default 2). Independently, :math:`V(\phi)` is evaluated on a dense
   0-360 deg grid (so oscillations *between* samples cannot hide) and
   capped at ``FFPOPT_DIHED_BARRIER_ABS`` kcal/mol (default 30). If ridge
   cannot meet the limit, :math:`K` is scaled. A flat or near-flat
   residual therefore cannot grow a 25 kcal fake well.

3. **Model selection on** ``nprim``.
   Always fitting :math:`n=1,2,3` is what creates the null space. Nested
   models :math:`k=0,1,\ldots,n_{\mathrm{prim}}` are scored with Gaussian
   AIC on the residual sum of squares. The smallest :math:`k` whose AIC
   is within ``FFPOPT_DIHED_AIC_WINDOW`` (default 2) of the best is kept.
   :math:`k=0` (no torsion correction) wins for a residual that is not a
   Fourier series; the type is stored as ``nprim=1`` with :math:`K=0`
   (leave the parent GAFF term). Disable with
   ``FFPOPT_DIHED_NPRIM_SELECT=0``.

4. **Amber-safety valve.**
   After the steps above, :math:`K` is clipped to
   ``+/- FFPOPT_DIHED_FC_MAX`` (default 25 kcal/mol). Logs
   ``Amber FC valve clipped ...`` only if this last clip actually hits.
   Extended ``--fit-full`` L-BFGS-B still uses the same box as optimizer
   bounds.

This is still **1-D scans per bond** and a **joint bytype shape-match**.
It is not an N-D coupled wavefront. Boltzmann-weighted residuals and a
GAFF-centered prior are possible later; they are not the default.

Logs (ASCII)
------------

::

   [ffpopt] nprim select c3-c3-s6-o: max=3 -> 1 (AIC=..., rss=..., window=2)
   [ffpopt] Fourier ridge at joint LS: kept 2/3 SVD modes, lambda=0, ...
   [ffpopt] energy-domain barrier joint LS dense: ... Vptp 48 -> 30 kcal
   [ffpopt] Amber FC valve clipped 1 dihedral FC(s) to +/-25 ...

Knobs live in ``ffpopt/pkgdata/files/env_defaults.json``. Implementation:
``ffpopt.dihed.DihedFitRegularize``.
