"""Collection of pre-built advection-diffusion kernels.

See `this tutorial <../examples/tutorial_diffusion.ipynb>`__ for a detailed explanation.
"""

import numpy as np

__all__ = ["AdvectionDiffusionEM", "AdvectionDiffusionM1", "DiffusionUniformKh"]


def meters_to_degrees_zonal(deg, lat, deg2m):  # pragma: no cover
    """Convert square meters to square degrees longitude at a given latitude."""
    return deg / pow(deg2m * np.cos(lat * np.pi / 180), 2)


def meters_to_degrees_meridional(deg, deg2m):  # pragma: no cover
    """Convert square meters to square degrees latitude."""
    return deg / pow(deg2m, 2)


def AdvectionDiffusionM1(particles, fieldset):  # pragma: no cover
    """Kernel for 2D advection-diffusion, solved using the Milstein scheme at first order (M1).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central
    difference gradient approximation. This should be (of the order of) the
    local gridsize.

    This Milstein scheme is of strong and weak order 1, which is higher than the
    Euler-Maruyama scheme. It experiences less spurious diffusivity by
    including extra correction terms that are computationally cheap.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))

    Kxp1 = fieldset.Kh_zonal[particles.t, particles.z, particles.y, particles.x + fieldset.dres, particles]
    Kxm1 = fieldset.Kh_zonal[particles.t, particles.z, particles.y, particles.x - fieldset.dres, particles]
    if fieldset.Kh_zonal.grid.mesh.is_spherical():
        Kxp1 = meters_to_degrees_zonal(Kxp1, particles.y, fieldset.Kh_zonal.grid.deg2m)
        Kxm1 = meters_to_degrees_zonal(Kxm1, particles.y, fieldset.Kh_zonal.grid.deg2m)
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)

    u, v = fieldset.UV[particles.t, particles.z, particles.y, particles.x, particles]
    kh_zonal = fieldset.Kh_zonal[particles.t, particles.z, particles.y, particles.x, particles]
    if fieldset.Kh_zonal.grid.mesh.is_spherical():
        kh_zonal = meters_to_degrees_zonal(kh_zonal, particles.y, fieldset.Kh_zonal.grid.deg2m)
    bx = np.sqrt(2 * kh_zonal)

    Kyp1 = fieldset.Kh_meridional[particles.t, particles.z, particles.y + fieldset.dres, particles.x, particles]
    Kym1 = fieldset.Kh_meridional[particles.t, particles.z, particles.y - fieldset.dres, particles.x, particles]
    if fieldset.Kh_meridional.grid.mesh.is_spherical():
        Kyp1 = meters_to_degrees_meridional(Kyp1, fieldset.Kh_meridional.grid.deg2m)
        Kym1 = meters_to_degrees_meridional(Kym1, fieldset.Kh_meridional.grid.deg2m)
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)

    kh_meridional = fieldset.Kh_meridional[particles.t, particles.z, particles.y, particles.x, particles]
    if fieldset.Kh_meridional.grid.mesh.is_spherical():
        kh_meridional = meters_to_degrees_meridional(kh_meridional, fieldset.Kh_meridional.grid.deg2m)
    by = np.sqrt(2 * kh_meridional)

    # Particle positions are updated only after evaluating all terms.
    particles.dx += u * particles.dt + 0.5 * dKdx * (dWx**2 + particles.dt) + bx * dWx
    particles.dy += v * particles.dt + 0.5 * dKdy * (dWy**2 + particles.dt) + by * dWy


def AdvectionDiffusionEM(particles, fieldset):  # pragma: no cover
    """Kernel for 2D advection-diffusion, solved using the Euler-Maruyama scheme (EM).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central
    difference gradient approximation. This should be (of the order of) the
    local gridsize.

    The Euler-Maruyama scheme is of strong order 0.5 and weak order 1.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))

    u, v = fieldset.UV[particles.t, particles.z, particles.y, particles.x, particles]

    Kxp1 = fieldset.Kh_zonal[particles.t, particles.z, particles.y, particles.x + fieldset.dres, particles]
    Kxm1 = fieldset.Kh_zonal[particles.t, particles.z, particles.y, particles.x - fieldset.dres, particles]
    if fieldset.Kh_zonal.grid.mesh.is_spherical():
        Kxp1 = meters_to_degrees_zonal(Kxp1, particles.y, fieldset.Kh_zonal.grid.deg2m)
        Kxm1 = meters_to_degrees_zonal(Kxm1, particles.y, fieldset.Kh_zonal.grid.deg2m)
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)
    ax = u + dKdx

    kh_zonal = fieldset.Kh_zonal[particles.t, particles.z, particles.y, particles.x, particles]
    if fieldset.Kh_zonal.grid.mesh.is_spherical():
        kh_zonal = meters_to_degrees_zonal(kh_zonal, particles.y, fieldset.Kh_zonal.grid.deg2m)
    bx = np.sqrt(2 * kh_zonal)

    Kyp1 = fieldset.Kh_meridional[particles.t, particles.z, particles.y + fieldset.dres, particles.x, particles]
    Kym1 = fieldset.Kh_meridional[particles.t, particles.z, particles.y - fieldset.dres, particles.x, particles]
    if fieldset.Kh_meridional.grid.mesh.is_spherical():
        Kyp1 = meters_to_degrees_meridional(Kyp1, fieldset.Kh_meridional.grid.deg2m)
        Kym1 = meters_to_degrees_meridional(Kym1, fieldset.Kh_meridional.grid.deg2m)
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)
    ay = v + dKdy

    kh_meridional = fieldset.Kh_meridional[particles.t, particles.z, particles.y, particles.x, particles]
    if fieldset.Kh_meridional.grid.mesh.is_spherical():
        kh_meridional = meters_to_degrees_meridional(kh_meridional, fieldset.Kh_meridional.grid.deg2m)
    by = np.sqrt(2 * kh_meridional)

    # Particle positions are updated only after evaluating all terms.
    particles.dx += ax * particles.dt + bx * dWx
    particles.dy += ay * particles.dt + by * dWy


def DiffusionUniformKh(particles, fieldset):  # pragma: no cover
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.

    Assumes that fieldset has constant fields `Kh_zonal` and `Kh_meridional`.
    These can be added via e.g.
    `fieldset.add_constant_field("Kh_zonal", kh_zonal)`
    or
    `fieldset.add_constant_field("Kh_meridional", kh_meridional)`
    where mesh is either 'flat' or 'spherical'

    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
    Since the perturbation due to diffusion is in this case isotropic independent, this
    kernel contains no advection and can be used in combination with a separate
    advection kernel.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))

    kh_zonal = fieldset.Kh_zonal[particles]
    kh_meridional = fieldset.Kh_meridional[particles]

    if fieldset.Kh_zonal.grid.mesh.is_spherical():
        kh_zonal = meters_to_degrees_zonal(kh_zonal, particles.y, fieldset.Kh_zonal.grid.deg2m)
        kh_meridional = meters_to_degrees_meridional(kh_meridional, fieldset.Kh_meridional.grid.deg2m)

    bx = np.sqrt(2 * kh_zonal)
    by = np.sqrt(2 * kh_meridional)

    particles.dx += bx * dWx
    particles.dy += by * dWy
