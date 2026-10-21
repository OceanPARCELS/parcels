from collections.abc import Callable
from typing import Literal

import numpy as np

from parcels._core.mesh import BaseMesh

__all__ = []


def phi1D_lin(xsi: float) -> list[float]:
    phi = [1 - xsi, xsi]
    return phi


# fmt: off
def phi1D_quad(xsi: float) -> list[float]:
    phi = [2*xsi**2-3*xsi+1,
           -4*xsi**2+4*xsi,
           2*xsi**2-xsi]

    return phi


def phi2D_lin(eta: float, xsi: float) -> np.ndarray:
    phi = np.column_stack([(1-xsi) * (1-eta),
                              xsi  * (1-eta),
                              xsi  *    eta ,
                           (1-xsi) *    eta ])

    return phi


def dphidxsi3D_lin(zeta: float, eta: float, xsi: float) -> tuple[list[float], list[float], list[float]]:
    dphidxsi = [ - (1-eta) * (1-zeta),
                   (1-eta) * (1-zeta),
                   (  eta) * (1-zeta),
                 - (  eta) * (1-zeta),
                 - (1-eta) * (  zeta),
                   (1-eta) * (  zeta),
                   (  eta) * (  zeta),
                 - (  eta) * (  zeta)]
    dphideta = [ - (1-xsi) * (1-zeta),
                 - (  xsi) * (1-zeta),
                   (  xsi) * (1-zeta),
                   (1-xsi) * (1-zeta),
                 - (1-xsi) * (  zeta),
                 - (  xsi) * (  zeta),
                   (  xsi) * (  zeta),
                   (1-xsi) * (  zeta)]
    dphidzet = [ - (1-xsi) * (1-eta),
                 - (  xsi) * (1-eta),
                 - (  xsi) * (  eta),
                 - (1-xsi) * (  eta),
                   (1-xsi) * (1-eta),
                   (  xsi) * (1-eta),
                   (  xsi) * (  eta),
                   (1-xsi) * (  eta)]

    return dphidxsi, dphideta, dphidzet


def dxdxsi3D_lin(
    hexa_z: list[float], hexa_y: list[float], hexa_x: list[float], zeta: float, eta: float, xsi: float, mesh: BaseMesh,
    deg2m: float = 1852 * 60.0
) -> tuple[float, float, float, float, float, float, float, float, float]:
    dphidxsi, dphideta, dphidzet = dphidxsi3D_lin(zeta, eta, xsi)

    if mesh.is_spherical():
        rad = np.pi / 180.
        lat = (1-xsi) * (1-eta) * hexa_y[0] + \
                 xsi  * (1-eta) * hexa_y[1] + \
                 xsi  *    eta  * hexa_y[2] + \
              (1-xsi) *    eta  * hexa_y[3]
        jac_lon = deg2m * np.cos(rad * lat)
        jac_lat = deg2m
    else:
        jac_lon = 1
        jac_lat = 1

    dxdxsi = np.dot(hexa_x, dphidxsi) * jac_lon
    dxdeta = np.dot(hexa_x, dphideta) * jac_lon
    dxdzet = np.dot(hexa_x, dphidzet) * jac_lon
    dydxsi = np.dot(hexa_y, dphidxsi) * jac_lat
    dydeta = np.dot(hexa_y, dphideta) * jac_lat
    dydzet = np.dot(hexa_y, dphidzet) * jac_lat
    dzdxsi = np.dot(hexa_z, dphidxsi)
    dzdeta = np.dot(hexa_z, dphideta)
    dzdzet = np.dot(hexa_z, dphidzet)

    return dxdxsi, dxdeta, dxdzet, dydxsi, dydeta, dydzet, dzdxsi, dzdeta, dzdzet


def jacobian3D_lin(
    hexa_z: list[float], hexa_y: list[float], hexa_x: list[float], zeta: float, eta: float, xsi: float, mesh: BaseMesh,
    deg2m: float = 1852 * 60.0
) -> float:
    dxdxsi, dxdeta, dxdzet, dydxsi, dydeta, dydzet, dzdxsi, dzdeta, dzdzet = dxdxsi3D_lin(hexa_z, hexa_y, hexa_x, zeta, eta, xsi, mesh, deg2m)

    jac = (
        dxdxsi * (dydeta * dzdzet - dzdeta * dydzet)
        - dxdeta * (dydxsi * dzdzet - dzdxsi * dydzet)
        + dxdzet * (dydxsi * dzdeta - dzdxsi * dydeta)
    )
    return jac


def jacobian3D_lin_face(
    hexa_z: list[float],
    hexa_y: list[float],
    hexa_x: list[float],
    zeta: float,
    eta: float,
    xsi: float,
    orientation: Literal["zonal", "meridional", "vertical"],
    mesh: BaseMesh,
) -> float:
    dxdxsi, dxdeta, dxdzet, dydxsi, dydeta, dydzet, dzdxsi, dzdeta, dzdzet = dxdxsi3D_lin(hexa_z, hexa_y, hexa_x, zeta, eta, xsi, mesh)

    if orientation == 'zonal':
        j = [dydeta*dzdzet-dydzet*dzdeta,
            -dxdeta*dzdzet+dxdzet*dzdeta,
             dxdeta*dydzet-dxdzet*dydeta]
    elif orientation == 'meridional':
        j = [dydxsi*dzdzet-dydzet*dzdxsi,
            -dxdxsi*dzdzet+dxdzet*dzdxsi,
             dxdxsi*dydzet-dxdzet*dydxsi]
    elif orientation == 'vertical':
        j = [dydxsi*dzdeta-dydeta*dzdxsi,
            -dxdxsi*dzdeta+dxdeta*dzdxsi,
             dxdxsi*dydeta-dxdeta*dydxsi]

    jac = np.sqrt(j[0]**2+j[1]**2+j[2]**2)
    return jac


def dphidxsi2D_lin(eta: float, xsi: float) -> tuple[list[float], list[float]]:
    dphidxsi = [-(1-eta),
                  1-eta,
                    eta,
                -   eta]
    dphideta = [-(1-xsi),
                -   xsi,
                    xsi,
                  1-xsi]

    return dphideta, dphidxsi
# fmt: on


def dxdxsi2D_lin(
    quad_y,
    quad_x,
    eta: float,
    xsi: float,
):
    dphideta, dphidxsi = dphidxsi2D_lin(eta, xsi)

    dxdxsi = np.dot(quad_x, dphidxsi)
    dxdeta = np.dot(quad_x, dphideta)
    dydxsi = np.dot(quad_y, dphidxsi)
    dydeta = np.dot(quad_y, dphideta)

    return dxdxsi, dxdeta, dydxsi, dydeta


def jacobian2D_lin(quad_y, quad_x, eta: float, xsi: float):
    dxdxsi, dxdeta, dydxsi, dydeta = dxdxsi2D_lin(quad_y, quad_x, eta, xsi)

    jac = dxdxsi * dydeta - dxdeta * dydxsi
    return jac


def interpolate(phi: Callable[[float], list[float]], f: list[float], xsi: float) -> float:
    return np.dot(phi(xsi), f)


def _geodetic_distance(
    lat1: float, lat2: float, lon1: float, lon2: float, mesh: BaseMesh, lat: float, deg2m: float = 1852 * 60.0
) -> float:
    if mesh.is_spherical():
        rad = np.pi / 180.0
        return np.sqrt(((lon2 - lon1) * deg2m * np.cos(rad * lat)) ** 2 + ((lat2 - lat1) * deg2m) ** 2)
    else:
        return np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)


def _compute_jacobian_determinant(py: np.ndarray, px: np.ndarray, eta: float, xsi: float) -> float:
    dphidxsi = np.column_stack([eta - 1, 1 - eta, eta, -eta])
    dphideta = np.column_stack([xsi - 1, -xsi, xsi, 1 - xsi])

    dxdxsi_diag = np.einsum("ij,ji->i", dphidxsi, px)
    dxdeta_diag = np.einsum("ij,ji->i", dphideta, px)
    dydxsi_diag = np.einsum("ij,ji->i", dphidxsi, py)
    dydeta_diag = np.einsum("ij,ji->i", dphideta, py)

    jac_diag = dxdxsi_diag * dydeta_diag - dxdeta_diag * dydxsi_diag
    return jac_diag
