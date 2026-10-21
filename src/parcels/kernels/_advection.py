"""Collection of pre-built advection kernels."""

import math

import numpy as np

from parcels._core.statuscodes import StatusCode
from parcels.kernels._integration import RK2, RK4

__all__ = [
    "AdvectionAnalytical",
    "AdvectionEE",
    "AdvectionRK2",
    "AdvectionRK2_3D",
    "AdvectionRK4",
    "AdvectionRK4_3D",
    "AdvectionRK45",
]


def rhs_uv(fieldset, t, z, y, x, particles):
    return fieldset.UV[t, z, y, x, particles]


def rhs_uvw(fieldset, t, z, y, x, particles):
    return fieldset.UVW[t, z, y, x, particles]


def AdvectionRK2(particles, fieldset):  # pragma: no cover
    """Advection of particles using second-order Runge-Kutta integration."""
    u, v = RK2(particles, fieldset, rhs_uv)
    particles.dx += u * particles.dt
    particles.dy += v * particles.dt


def AdvectionRK2_3D(particles, fieldset):  # pragma: no cover
    """Advection of particles using second-order Runge-Kutta integration including vertical velocity."""
    u, v, w = RK2(particles, fieldset, rhs_uvw)
    particles.dx += u * particles.dt
    particles.dy += v * particles.dt
    particles.dz += w * particles.dt


def AdvectionRK4(particles, fieldset):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration."""
    u, v = RK4(particles, fieldset, rhs_uv)
    particles.dx += u * particles.dt
    particles.dy += v * particles.dt


def AdvectionRK4_3D(particles, fieldset):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity."""
    u, v, w = RK4(particles, fieldset, rhs_uvw)
    particles.dx += u * particles.dt
    particles.dy += v * particles.dt
    particles.dz += w * particles.dt


def AdvectionEE(particles, fieldset):  # pragma: no cover
    """Advection of particles using Explicit Euler (aka Euler Forward) integration."""
    (u1, v1) = fieldset.UV[particles]
    particles.dx += u1 * particles.dt
    particles.dy += v1 * particles.dt


def AdvectionRK45(particles, fieldset):  # pragma: no cover
    """Advection of particles using adaptive Runge-Kutta 4/5 integration.

    Note that this kernel requires a FieldSet with constants 'RK45_tol' (in meters),
    'RK45_min_dt' (in seconds) and 'RK45_max_dt' (in seconds).

    Time-step dt is halved if error is larger than fieldset.RK45_tol,
    and doubled if error is smaller than 1/10th of tolerance.
    """
    sign_dt = np.sign(particles.dt)

    c = [1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0]
    A = [
        [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0],
    ]
    b4 = [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0]
    b5 = [16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0]

    (u1, v1) = fieldset.UV[particles]
    x1 = particles.x + u1 * A[0][0] * particles.dt
    y1 = particles.y + v1 * A[0][0] * particles.dt
    (u2, v2) = fieldset.UV[particles.t + c[0] * particles.dt, particles.z, y1, x1, particles]
    x2 = particles.x + (u1 * A[1][0] + u2 * A[1][1]) * particles.dt
    y2 = particles.y + (v1 * A[1][0] + v2 * A[1][1]) * particles.dt
    (u3, v3) = fieldset.UV[particles.t + c[1] * particles.dt, particles.z, y2, x2, particles]
    x3 = particles.x + (u1 * A[2][0] + u2 * A[2][1] + u3 * A[2][2]) * particles.dt
    y3 = particles.y + (v1 * A[2][0] + v2 * A[2][1] + v3 * A[2][2]) * particles.dt
    (u4, v4) = fieldset.UV[particles.t + c[2] * particles.dt, particles.z, y3, x3, particles]
    x4 = particles.x + (u1 * A[3][0] + u2 * A[3][1] + u3 * A[3][2] + u4 * A[3][3]) * particles.dt
    y4 = particles.y + (v1 * A[3][0] + v2 * A[3][1] + v3 * A[3][2] + v4 * A[3][3]) * particles.dt
    (u5, v5) = fieldset.UV[particles.t + c[3] * particles.dt, particles.z, y4, x4, particles]
    x5 = particles.x + (u1 * A[4][0] + u2 * A[4][1] + u3 * A[4][2] + u4 * A[4][3] + u5 * A[4][4]) * particles.dt
    y5 = particles.y + (v1 * A[4][0] + v2 * A[4][1] + v3 * A[4][2] + v4 * A[4][3] + v5 * A[4][4]) * particles.dt
    (u6, v6) = fieldset.UV[particles.t + c[4] * particles.dt, particles.z, y5, x5, particles]

    x_4th = (u1 * b4[0] + u2 * b4[1] + u3 * b4[2] + u4 * b4[3] + u5 * b4[4]) * particles.dt
    y_4th = (v1 * b4[0] + v2 * b4[1] + v3 * b4[2] + v4 * b4[3] + v5 * b4[4]) * particles.dt
    x_5th = (u1 * b5[0] + u2 * b5[1] + u3 * b5[2] + u4 * b5[3] + u5 * b5[4] + u6 * b5[5]) * particles.dt
    y_5th = (v1 * b5[0] + v2 * b5[1] + v3 * b5[2] + v4 * b5[3] + v5 * b5[4] + v6 * b5[5]) * particles.dt

    kappa = np.sqrt(np.pow(x_5th - x_4th, 2) + np.pow(y_5th - y_4th, 2))

    good_particles = (kappa <= fieldset.RK45_tol) | (np.fabs(particles.dt) <= np.fabs(fieldset.RK45_min_dt))
    particles.dx += np.where(good_particles, x_5th, 0)
    particles.dy += np.where(good_particles, y_5th, 0)

    increase_dt_particles = (
        good_particles
        & (kappa <= fieldset.RK45_tol / 10)
        & (np.fabs(particles.dt * 2) <= np.fabs(fieldset.RK45_max_dt))
    )
    particles.next_dt = np.where(increase_dt_particles, particles.dt * 2, particles.dt)
    particles.next_dt = np.where(
        np.abs(particles.next_dt) > np.abs(fieldset.RK45_max_dt),
        fieldset.RK45_max_dt * sign_dt,
        particles.next_dt,
    )
    particles.state = np.where(good_particles, StatusCode.Success, particles.state)

    repeat_particles = np.invert(good_particles)
    particles.dt = np.where(repeat_particles, particles.dt / 2, particles.dt)
    particles.dt = np.where(
        np.abs(particles.dt) < np.abs(fieldset.RK45_min_dt),
        fieldset.RK45_min_dt * sign_dt,
        particles.dt,
    )
    particles.state = np.where(repeat_particles, StatusCode.Repeat, particles.state)


def AdvectionAnalytical(particles, fieldset):  # pragma: no cover
    """Advection of particles using 'analytical advection' integration.

    Based on Ariane/TRACMASS algorithm, as detailed in e.g. Doos et al (https://doi.org/10.5194/gmd-10-1733-2017).
    Note that the time-dependent scheme is currently implemented with 'intermediate timesteps'
    (default 10 per model timestep) and not yet with the full analytical time integration.
    """
    import numpy as np

    import parcels._core.utils.interpolation as i_u

    tol = 1e-10
    I_s = 10  # number of intermediate time steps
    dt = particles.dt
    direction = 1.0 if dt > 0 else -1.0
    withW = True if "W" in [f.name for f in fieldset.fields.values()] else False
    withTime = True if len(fieldset.U.grid.time) > 1 else False
    tau, zeta, eta, xsi, ti, zi, yi, xi = fieldset.U._search_indices(
        particles.z, particles.y, particles.x, particles=particles
    )
    ds_t = dt
    if withTime:
        time_i = np.linspace(0, fieldset.U.grid.time[ti + 1] - fieldset.U.grid.time[ti], I_s)
        ds_t = min(ds_t, time_i[np.where(particles.time - fieldset.U.grid.time[ti] < time_i)[0][0]])

    if withW:
        if abs(xsi - 1) < tol:
            if fieldset.U.data[0, zi + 1, yi + 1, xi + 1] > 0:
                xi += 1
                xsi = 0
        if abs(eta - 1) < tol:
            if fieldset.V.data[0, zi + 1, yi + 1, xi + 1] > 0:
                yi += 1
                eta = 0
        if abs(zeta - 1) < tol:
            if fieldset.W.data[0, zi + 1, yi + 1, xi + 1] > 0:
                zi += 1
                zeta = 0
    else:
        if abs(xsi - 1) < tol:
            if fieldset.U.data[0, yi + 1, xi + 1] > 0:
                xi += 1
                xsi = 0
        if abs(eta - 1) < tol:
            if fieldset.V.data[0, yi + 1, xi + 1] > 0:
                yi += 1
                eta = 0

    particles.ei[:] = fieldset.U.ravel_index(zi, yi, xi)

    grid = fieldset.U.grid
    if grid._gtype < 2:
        px = np.array([grid.x[xi], grid.x[xi + 1], grid.x[xi + 1], grid.x[xi]])
        py = np.array([grid.y[yi], grid.y[yi], grid.y[yi + 1], grid.y[yi + 1]])
    else:
        px = np.array([grid.x[yi, xi], grid.x[yi, xi + 1], grid.x[yi + 1, xi + 1], grid.x[yi + 1, xi]])
        py = np.array([grid.y[yi, xi], grid.y[yi, xi + 1], grid.y[yi + 1, xi + 1], grid.y[yi + 1, xi]])
    if grid.mesh == "spherical":
        px[0] = px[0] + 360 if px[0] < particles.x - 225 else px[0]
        px[0] = px[0] - 360 if px[0] > particles.y + 225 else px[0]
        px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
        px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
    if withW:
        pz = np.array([grid.depth[zi], grid.depth[zi + 1]])
        dz = pz[1] - pz[0]
    else:
        dz = 1.0

    c1 = i_u._geodetic_distance(py[0], py[1], px[0], px[1], grid.mesh, np.dot(i_u.phi2D_lin(0.0, xsi), py), grid.deg2m)
    c2 = i_u._geodetic_distance(py[1], py[2], px[1], px[2], grid.mesh, np.dot(i_u.phi2D_lin(eta, 1.0), py), grid.deg2m)
    c3 = i_u._geodetic_distance(py[2], py[3], px[2], px[3], grid.mesh, np.dot(i_u.phi2D_lin(1.0, xsi), py), grid.deg2m)
    c4 = i_u._geodetic_distance(py[3], py[0], px[3], px[0], grid.mesh, np.dot(i_u.phi2D_lin(eta, 0.0), py), grid.deg2m)
    rad = np.pi / 180.0
    deg2m = grid.deg2m
    meshJac = (deg2m * deg2m * math.cos(rad * particles.y)) if grid.mesh == "spherical" else 1
    dxdy = i_u._compute_jacobian_determinant(py, px, eta, xsi) * meshJac

    if withW:
        U0 = direction * fieldset.U.data[ti, zi + 1, yi + 1, xi] * c4 * dz
        U1 = direction * fieldset.U.data[ti, zi + 1, yi + 1, xi + 1] * c2 * dz
        V0 = direction * fieldset.V.data[ti, zi + 1, yi, xi + 1] * c1 * dz
        V1 = direction * fieldset.V.data[ti, zi + 1, yi + 1, xi + 1] * c3 * dz
        if withTime:
            U0 = U0 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, zi + 1, yi + 1, xi] * c4 * dz
            U1 = U1 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, zi + 1, yi + 1, xi + 1] * c2 * dz
            V0 = V0 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, zi + 1, yi, xi + 1] * c1 * dz
            V1 = V1 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, zi + 1, yi + 1, xi + 1] * c3 * dz
    else:
        U0 = direction * fieldset.U.data[ti, yi + 1, xi] * c4 * dz
        U1 = direction * fieldset.U.data[ti, yi + 1, xi + 1] * c2 * dz
        V0 = direction * fieldset.V.data[ti, yi, xi + 1] * c1 * dz
        V1 = direction * fieldset.V.data[ti, yi + 1, xi + 1] * c3 * dz
        if withTime:
            U0 = U0 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, yi + 1, xi] * c4 * dz
            U1 = U1 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, yi + 1, xi + 1] * c2 * dz
            V0 = V0 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, yi, xi + 1] * c1 * dz
            V1 = V1 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, yi + 1, xi + 1] * c3 * dz

    def compute_ds(F0, F1, r, direction, tol):  # noqa: N803
        up = F0 * (1 - r) + F1 * r
        r_target = 1.0 if direction * up >= 0.0 else 0.0
        B = F0 - F1
        delta = -F0
        B = 0 if abs(B) < tol else B

        if abs(B) > tol:
            F_r1 = r_target + delta / B
            F_r0 = r + delta / B
        else:
            F_r0, F_r1 = None, None

        if abs(B) < tol and abs(delta) < tol:
            ds = float("inf")
        elif B == 0:
            ds = -(r_target - r) / delta
        elif F_r1 * F_r0 < tol:
            ds = float("inf")
        else:
            ds = -1.0 / B * math.log(F_r1 / F_r0)

        if abs(ds) < tol:
            ds = float("inf")
        return ds, B, delta

    ds_x, B_x, delta_x = compute_ds(U0, U1, xsi, direction, tol)
    ds_y, B_y, delta_y = compute_ds(V0, V1, eta, direction, tol)
    if withW:
        W0 = direction * fieldset.W.data[ti, zi, yi + 1, xi + 1] * dxdy
        W1 = direction * fieldset.W.data[ti, zi + 1, yi + 1, xi + 1] * dxdy
        if withTime:
            W0 = W0 * (1 - tau) + tau * direction * fieldset.W.data[ti + 1, zi, yi + 1, xi + 1] * dxdy
            W1 = W1 * (1 - tau) + tau * direction * fieldset.W.data[ti + 1, zi + 1, yi + 1, xi + 1] * dxdy
        ds_z, B_z, delta_z = compute_ds(W0, W1, zeta, direction, tol)
    else:
        ds_z = float("inf")

    # take the minimum travel time
    s_min = min(abs(ds_x), abs(ds_y), abs(ds_z), abs(ds_t / (dxdy * dz)))

    # calculate end position in time s_min
    def compute_rs(r, B, delta, s_min):  # noqa: N803
        if abs(B) < tol:
            return -delta * s_min + r
        else:
            return (r + delta / B) * math.exp(-B * s_min) - delta / B

    rs_x = compute_rs(xsi, B_x, delta_x, s_min)
    rs_y = compute_rs(eta, B_y, delta_y, s_min)

    particles.dx += (
        (1.0 - rs_x) * (1.0 - rs_y) * px[0]
        + rs_x * (1.0 - rs_y) * px[1]
        + rs_x * rs_y * px[2]
        + (1.0 - rs_x) * rs_y * px[3]
        - particles.x
    )
    particles.dy += (
        (1.0 - rs_x) * (1.0 - rs_y) * py[0]
        + rs_x * (1.0 - rs_y) * py[1]
        + rs_x * rs_y * py[2]
        + (1.0 - rs_x) * rs_y * py[3]
        - particles.y
    )

    if withW:
        rs_z = compute_rs(zeta, B_z, delta_z, s_min)
        particles.dz += (1.0 - rs_z) * pz[0] + rs_z * pz[1] - particles.z

    if particles.dt > 0:
        particles.dt = max(direction * s_min * (dxdy * dz), 1e-7).astype("timedelta64[s]")
    else:
        particles.dt = min(direction * s_min * (dxdy * dz), -1e-7).astype("timedelta64[s]")
