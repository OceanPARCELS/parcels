"""Collection of time integrators for use in Parcels Kernels"""


def _validate_rhs_output(output, integrator_name):
    """Validate that an rhs function returned a 2- or 3-component tuple (u, v) or (u, v, w)."""
    if not isinstance(output, tuple):
        raise TypeError(
            f"{integrator_name}: rhs must return a tuple of (u, v) or (u, v, w), got {type(output).__name__}."
        )
    if len(output) not in (2, 3):
        raise ValueError(
            f"{integrator_name}: rhs must return a tuple of 2 (u, v) or 3 (u, v, w) components, got {len(output)}."
        )


def RK2(particles, fieldset, rhs):
    z, y, x = particles.z, particles.y, particles.x
    fields = rhs(fieldset, particles.t, z, y, x, particles)
    _validate_rhs_output(fields, "RK2")
    x = particles.x + fields[0] * 0.5 * particles.dt
    y = particles.y + fields[1] * 0.5 * particles.dt
    if len(fields) > 2:
        z = particles.z + fields[2] * 0.5 * particles.dt
    t = particles.t + 0.5 * particles.dt
    return rhs(fieldset, t, z, y, x, particles)


def RK4(particles, fieldset, rhs):
    z, y, x = particles.z, particles.y, particles.x
    k1 = rhs(fieldset, particles.t, z, y, x, particles)
    _validate_rhs_output(k1, "RK4")
    x = particles.x + k1[0] * 0.5 * particles.dt
    y = particles.y + k1[1] * 0.5 * particles.dt
    if len(k1) > 2:
        z = particles.z + k1[2] * 0.5 * particles.dt
    t = particles.t + 0.5 * particles.dt
    k2 = rhs(fieldset, t, z, y, x, particles)
    x = particles.x + k2[0] * 0.5 * particles.dt
    y = particles.y + k2[1] * 0.5 * particles.dt
    if len(k2) > 2:
        z = particles.z + k2[2] * 0.5 * particles.dt
    k3 = rhs(fieldset, t, z, y, x, particles)
    x = particles.x + k3[0] * particles.dt
    y = particles.y + k3[1] * particles.dt
    if len(k3) > 2:
        z = particles.z + k3[2] * particles.dt
    t = particles.t + particles.dt
    k4 = rhs(fieldset, t, z, y, x, particles)
    if len(k4) == 2:
        return ((k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0, (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0)
    if len(k4) == 3:
        return (
            (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0,
            (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0,
            (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0,
        )
