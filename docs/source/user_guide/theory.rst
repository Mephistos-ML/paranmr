.. _theory:

Theory
======

Below, we outline the key models and equations implemented in ``simpnmr``.

.. _TOTAL:

Total shift
-----------

In ``simpnmr``, the FC, PCS, and orbital terms are treated as separate shift
contributions. The total predicted chemical shift is therefore written as

.. math::
   :label: :eq: total_orb

    \delta^{\mathrm{TOTAL}}=\delta^{\mathrm{DIA}}+\delta^{\mathrm{FC}}+\delta^{\mathrm{PCS}}+\delta^{\mathrm{ORB}}

and as

.. math::
   :label: :eq: total_no_orb

    \delta^{\mathrm{TOTAL}}=\delta^{\mathrm{DIA}}+\delta^{\mathrm{FC}}+\delta^{\mathrm{PCS}}

when no orbital contribution is available.

.. note::

    The calculation without :math:`\delta^{\mathrm{ORB}}` remains physically reasonable
    because the orbital contribution is small and expected to become negligible at sufficiently
    large distances from the paramagnetic centre.


.. _DIA:

Diamagnetic shift
-----------------

When the diamagnetic contribution is obtained from DFT shielding data, a
reference value is required in order to convert shielding into a chemical
shift. In ``simpnmr``, the diamagnetic shift contribution is written as

.. math::
   :label: :eq: dia

    \delta^{\mathrm{DIA}}=\sigma_{\mathrm{ref}}-\sigma

where :math:`\sigma` is the calculated shielding for the nucleus of interest
and :math:`\sigma_{\mathrm{ref}}` is the corresponding reference shielding.

This diamagnetic term is then added directly to the other shift contributions
when forming :math:`\delta^{\mathrm{TOTAL}}`.

.. note::

   For a consistent diamagnetic shift, the calculated shielding
   :math:`\sigma` and the reference shielding :math:`\sigma_{\mathrm{ref}}`
   must be obtained at the same level of theory.

.. _PCS:

Pseudocontact shift
-------------------
Pseudocontact shift (PCS) depends on the anisotropic part of the magnetic
susceptibility tensor and the spin-dipolar part of the hyperfine coupling
tensor (HFC).

1. PCS with hyperfine from DFT 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To account for the non-point nature of the paramagnetic centre, the normalised
traceless spin-dipolar hyperfine contribution can be obtained from a simple
single-point DFT calculation of :math:`\mathbf{A}^{\mathrm{SD}}`:

.. math::
   :label: :eq: pcs

   \delta^{\mathrm{PCS}}=\frac{1}{3} \operatorname{tr}\left(\Delta \boldsymbol{\chi} \cdot \mathbf{A}^{\mathrm{SD}}\right)

In ``simpnmr``, this defines the PCS contribution independently of any
isotropic and orbital shift terms.

.. note::

   The spin-dipolar hyperfine tensor :math:`\mathbf{A}^{\mathrm{SD}}` is
   always traceless.

2. PCS with point-dipole approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming that a paramagnetic metal centre is at the origin and a nucleus of
interest has coordinates :math:`(x, y, z)`, the pseudocontact shift
contribution :math:`\delta^{\mathrm{PCS}}` can be calculated as a third of the
trace of the magnetic susceptibility tensor :math:`\chi` multiplied by the
traceless spin-dipolar hyperfine tensor, which in the point-dipole
approximation is a matrix that depends only on the nuclear coordinates:

.. math::
   :label: :eq: pcs_pd

    \delta^{\mathrm{PCS}}=\frac{1}{12 \pi r^5} \operatorname{tr}\left[\left(\begin{array}{ccc}
    \chi_{x x} & \chi_{x y} & \chi_{x z} \\
    \chi_{y x} & \chi_{y y} & \chi_{y z} \\
    \chi_{z x} & \chi_{z y} & \chi_{z z}
    \end{array}\right) \cdot\left(\begin{array}{ccc}
    3 x^2-r^2 & 3 x y & 3 x z \\
    3 x y & 3 y^2-r^2 & 3 y z \\
    3 x z & 3 y z & 3 z^2-r^2
    \end{array}\right)\right]

If the coordinates are specified in Å and :math:`\chi` is in Å\ :sup:`3`,
then the equation above, multiplied by 10\ :sup:`6`, gives the PCS in ppm.

.. _FC:

Fermi-contact shift
-------------------

The Fermi-contact shift contribution :math:`\delta^{\mathrm{FC}}` is evaluated
from the isotropic Fermi-contact hyperfine interaction at the nucleus together
with the corresponding isotropic magnetic susceptibility term.

.. note::

   By construction, the Fermi-contact hyperfine tensor
   :math:`\mathbf{A}^{\mathrm{FC}}` is isotropic. In matrix form, it is
   diagonal with equal diagonal elements.

1. FC with spin-only magnetic susceptibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the simplest model, the Fermi-contact shift is proportional to the
isotropic Fermi-contact hyperfine interaction at the nucleus of interest and
the spin-only magnetic susceptibility:

.. math::
   :label: :eq: fc_s

    \delta^{\mathrm{FC}}=\chi_{iso}^S A^{FC}

In ``simpnmr``, :math:`\delta^{\mathrm{FC}}` is evaluated from the isotropic
part of the Fermi-contact hyperfine tensor, i.e. from
:math:`\frac{1}{3}\operatorname{tr}(\mathbf{A}^{\mathrm{FC}})`, where the spin-only magnetic susceptibility in SI units is:

.. math::
   :label: :eq: chi_s

    \chi_{iso}^S=\frac{\mu_0 \mu_B^2 \mathrm{g}_{\mathrm{e}}^2 S(S+1)}{3 k T}

where :math:`\mu_0` is the vacuum permeability, :math:`\mu_B` is the Bohr
magneton, :math:`\mathrm{g}_{\mathrm{e}}` is the free-electron g-factor,
:math:`S` is the total spin, :math:`k` is the Boltzmann constant, and
:math:`T` is the temperature.

2. FC with g-corrected magnetic susceptibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to account for the effect of :math:`\mathbf{g}_{\mathrm{ab-initio}}`
anisotropy on the FC shift, a g-corrected isotropic susceptibility is used.
The magnetic susceptibility tensor and the corresponding
:math:`\mathbf{g}_{\mathrm{ab-initio}}` tensor must be calculated at the same
level of theory (e.g. SOC-NEVPT2):

.. math::
    :label: :eq: FC_g

    \delta^{\mathrm{FC}}=\chi^{\mathrm{g-corr}}_{\mathrm{iso}}\,\frac{1}{3}\operatorname{tr}\left(\mathbf{A}^{\mathrm{FC}}\right)

where the g-corrected isotropic susceptibility is

.. math::
   :label: :eq: chi_g_corr

    \chi^{\mathrm{g-corr}}_{\mathrm{iso}}=\frac{g_{\mathrm{e}}}{3}\left(\frac{\chi_x}{g_x}+\frac{\chi_y}{g_y}+\frac{\chi_z}{g_z}\right)

.. note::

   Here, :math:`\mathbf{g}_{\mathrm{ab-initio}}` should be taken from the same
   level of theory as the susceptibility tensor used to compute
   :math:`\chi^{\mathrm{g-corr}}_{\mathrm{iso}}`.

.. _ORB:

Orbital shift contribution
--------------------------

In ``simpnmr``, the orbital contribution is treated as an additional shift
channel. It does not modify the definitions of :math:`\delta^{\mathrm{FC}}` or
:math:`\delta^{\mathrm{PCS}}`.

.. note::

   The orbital contribution is evaluated only when both of the following are
   available from the same QC source:

   - the orbital hyperfine contribution :math:`\mathbf{A}^{\mathrm{ORB}}`
   - the associated :math:`\mathbf{g}_{\mathrm{DFT}}` tensor

If the orbital hyperfine contribution :math:`\mathbf{A}^{\mathrm{ORB}}` and the
associated :math:`\mathbf{g}_{\mathrm{DFT}}` tensor are available from the
same QC source, the isotropic orbital contribution is evaluated as

.. math::
   :label: :eq: orb_iso

    \delta^{\mathrm{ORB}}_{\mathrm{iso}}=
    \chi_{\mathrm{iso}}\frac{1}{3}\operatorname{tr}\left[\frac{g_{\mathrm{e}}}{\mathbf{g}^{\mathrm{T}}_{\mathrm{DFT}}}\left(\mathbf{A}^{\mathrm{SD}}+\mathbf{A}^{\mathrm{ORB}}\right)^{\mathrm{T}}\right]

and the anisotropic orbital contribution is evaluated as

.. math::
   :label: :eq: orb_aniso

    \delta^{\mathrm{ORB}}_{\mathrm{aniso}}=\frac{1}{3}\operatorname{tr}\left[\Delta\boldsymbol{\chi}\frac{g_{\mathrm{e}}}{\mathbf{g}^{\mathrm{T}}_{\mathrm{DFT}}}\left(\mathbf{A}^{\mathrm{SD}}+\mathbf{A}^{\mathrm{ORB}}\right)^{\mathrm{T}}-\Delta\boldsymbol{\chi}\mathbf{A}^{\mathrm{SD}}\right]

The total orbital shift contribution reported by ``simpnmr`` is the sum

.. math::
   :label: :eq: orb_total

    \delta^{\mathrm{ORB}}=\delta^{\mathrm{ORB}}_{\mathrm{iso}}+\delta^{\mathrm{ORB}}_{\mathrm{aniso}}

Therefore, evaluating orbital shift contributions requires both the orbital
hyperfine contribution and the associated :math:`\mathbf{g}_{\mathrm{DFT}}`
tensor from the same QC source.

.. note::

   Another common source of confusion is the role of
   :math:`\mathbf{A}^{\mathrm{SD}}` in the orbital expressions above. In
   ``simpnmr``, :math:`\mathbf{A}^{\mathrm{SD}}` still defines the PCS term on
   its own, while the orbital term is evaluated separately from the transformed
   combination :math:`\mathbf{A}^{\mathrm{SD}}+\mathbf{A}^{\mathrm{ORB}}`.

.. _PRE:

Paramagnetic relaxation enhancement
-----------------------------------
