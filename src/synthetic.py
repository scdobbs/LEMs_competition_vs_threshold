"""
Synthetic planform solvers for Peclet-number analysis.

Provides functions for building forced-routing V-shaped planform grids
and solving the steady-state advection-diffusion PDE on them.  Used in
resolution-dependence studies (supplementary figure 5).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Planform grid construction
# ---------------------------------------------------------------------------

def compute_area_planform(
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    A_top: float = 0.0,
) -> np.ndarray:
    """Build a synthetic V-shaped planform drainage area grid.

    Flow routes inward from left and right edges toward the centre column,
    then flows southward (increasing *j*) along the centre.

    Parameters
    ----------
    Nx : int
        Number of columns (must be odd).
    Ny : int
        Number of rows.
    dx, dy : float
        Cell spacing.
    A_top : float
        Extra upstream area injected at the top of the centre column
        (default 0).

    Returns
    -------
    np.ndarray
        ``(Ny, Nx)`` drainage area grid.

    Raises
    ------
    ValueError
        If *Nx* is even.
    """
    if Nx % 2 == 0:
        raise ValueError("Nx must be odd.")

    A = np.zeros((Ny, Nx), dtype=float)
    ci = (Nx - 1) // 2
    cellA = dx * dy

    for j in range(Ny):
        # Left side: accumulate left → centre
        acc = 0.0
        for i in range(ci):
            acc += cellA
            A[j, i] = acc

        # Right side: accumulate right → centre
        acc = 0.0
        for i in range(Nx - 1, ci, -1):
            acc += cellA
            A[j, i] = acc

        left_contrib = A[j, ci - 1] if ci - 1 >= 0 else 0.0
        right_contrib = A[j, ci + 1] if ci + 1 < Nx else 0.0

        if j == 0:
            A[j, ci] = A_top + left_contrib + right_contrib
        else:
            A[j, ci] = A[j - 1, ci] + left_contrib + right_contrib

    return A


def downstream_link(
    Nx: int,
    Ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D8 flow routing on forced V-shaped planform.

    All flow routes inward to the centre column, then southward to the
    outlet at ``(ci, Ny-1)``.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions (*Nx* must be odd).

    Returns
    -------
    i_dn : np.ndarray
        Column index of downstream neighbour (−1 if outlet).
    j_dn : np.ndarray
        Row index of downstream neighbour (−1 if outlet).
    is_outlet : np.ndarray
        Boolean mask (``True`` only at the outlet cell).
    """
    ci = (Nx - 1) // 2
    i_dn = np.full((Ny, Nx), -1, dtype=int)
    j_dn = np.full((Ny, Nx), -1, dtype=int)
    is_outlet = np.zeros((Ny, Nx), dtype=bool)

    for j in range(Ny):
        for i in range(Nx):
            if i == ci and j == Ny - 1:
                is_outlet[j, i] = True
                continue
            if i < ci:
                i_dn[j, i], j_dn[j, i] = i + 1, j
            elif i > ci:
                i_dn[j, i], j_dn[j, i] = i - 1, j
            else:
                i_dn[j, i], j_dn[j, i] = i, j + 1

    return i_dn, j_dn, is_outlet


# ---------------------------------------------------------------------------
# Laplacian stencil helpers
# ---------------------------------------------------------------------------

def _laplacian_reflecting_terms(
    i: int,
    j: int,
    idx_fn,
    Nx: int,
    Ny: int,
    inv_dx2: float,
    inv_dy2: float,
) -> list[tuple[int, float]]:
    """Build Laplacian stencil with reflecting (no-flux) boundary conditions.

    Parameters
    ----------
    i, j : int
        Cell column and row.
    idx_fn : callable
        ``idx(i, j) → flat_index``.
    Nx, Ny : int
        Grid dimensions.
    inv_dx2, inv_dy2 : float
        Pre-computed ``1/dx²`` and ``1/dy²``.

    Returns
    -------
    list of (col_index, coefficient)
        Entries to add to the sparse matrix row.
    """
    def reflect(ii: int, n: int) -> int:
        if ii < 0:
            return 1
        if ii >= n:
            return n - 2
        return ii

    ip = reflect(i + 1, Nx)
    im = reflect(i - 1, Nx)
    jp = reflect(j + 1, Ny)
    jm = reflect(j - 1, Ny)

    return [
        (idx_fn(i, j), -2.0 * (inv_dx2 + inv_dy2)),
        (idx_fn(ip, j), inv_dx2),
        (idx_fn(im, j), inv_dx2),
        (idx_fn(i, jp), inv_dy2),
        (idx_fn(i, jm), inv_dy2),
    ]


# ---------------------------------------------------------------------------
# PDE solver
# ---------------------------------------------------------------------------

def solve_advection_diffusion_planform(
    U: float,
    K: float,
    m: float,
    D: float,
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    A_top: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve steady-state advection-diffusion on a synthetic planform.

    Solves

    .. math::

        0 = U - K A^m S + D \\nabla^2 z

    on a forced V-shaped routing planform with no-flux lateral BCs and
    z = 0 at the outlet.

    Parameters
    ----------
    U : float
        Uplift rate.
    K : float
        Erodibility coefficient.
    m : float
        Area exponent.
    D : float
        Diffusivity.
    Nx, Ny : int
        Grid dimensions (*Nx* must be odd, both >= 5).
    Lx, Ly : float
        Domain dimensions in metres.
    A_top : float
        Extra upstream area at the top of the centre column (default 0).

    Returns
    -------
    Z : np.ndarray
        ``(Ny, Nx)`` steady-state elevation.
    A : np.ndarray
        ``(Ny, Nx)`` drainage area grid.
    dx : float
        Cell spacing in x.
    dy : float
        Cell spacing in y.

    Raises
    ------
    ValueError
        If *Nx* is even or dimensions are too small (< 5).
    """
    if Nx % 2 == 0:
        raise ValueError("Nx must be odd.")
    if Nx < 5 or Ny < 5:
        raise ValueError("Use Nx, Ny >= 5.")

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    ci = (Nx - 1) // 2

    A = compute_area_planform(Nx, Ny, dx, dy, A_top=A_top)
    i_dn, j_dn, is_outlet = downstream_link(Nx, Ny)

    Ntot = Nx * Ny

    def idx(ii: int, jj: int) -> int:
        return ii + Nx * jj

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    M = lil_matrix((Ntot, Ntot), dtype=float)
    b = np.zeros(Ntot, dtype=float)

    for j in range(Ny):
        for i in range(Nx):
            r = idx(i, j)

            if is_outlet[j, i]:
                M[r, r] = 1.0
                b[r] = 0.0
                continue

            # Diffusion (Laplacian with reflecting BCs)
            for c, coeff in _laplacian_reflecting_terms(
                i, j, idx, Nx, Ny, inv_dx2, inv_dy2,
            ):
                M[r, c] += D * coeff

            # Incision along forced routing
            idn, jdn = i_dn[j, i], j_dn[j, i]
            dL = dx if i != ci else dy
            fac = K * (A[j, i] ** m) / dL

            M[r, r] += -fac
            M[r, idx(idn, jdn)] += fac

            b[r] = -U

    z_flat = spsolve(M.tocsr(), b)
    Z = z_flat.reshape((Ny, Nx))

    return Z, A, dx, dy


# ---------------------------------------------------------------------------
# Peclet-number utilities
# ---------------------------------------------------------------------------

def inlet_area_from_Pe(
    Pe: float,
    D: float,
    K: float,
    L_PE: float,
    m: float,
) -> float:
    """Compute inlet area from Peclet number.

    .. math::

        A_{\\text{in}} = \\left( \\frac{Pe \\, D}{K \\, L} \\right)^{1/m}

    Parameters
    ----------
    Pe : float
        Peclet number.
    D, K : float
        Diffusivity and erodibility.
    L_PE : float
        Peclet length scale.
    m : float
        Area exponent.

    Returns
    -------
    float
        Inlet drainage area.
    """
    return float((Pe * D / (K * L_PE)) ** (1.0 / m))


def D_from_Pe(
    Pe: float,
    A_inlet: float,
    K: float,
    L_PE: float,
    m: float,
) -> float:
    """Compute diffusivity from Peclet number and inlet area.

    Parameters
    ----------
    Pe, A_inlet, K, L_PE, m : float

    Returns
    -------
    float
    """
    return float((K * L_PE * (A_inlet ** m)) / Pe)


def Pe_from_D_Ain(
    D: float,
    A_inlet: float,
    K: float,
    L_PE: float,
    m: float,
) -> float:
    """Compute Peclet number from diffusivity and inlet area.

    Parameters
    ----------
    D, A_inlet, K, L_PE, m : float

    Returns
    -------
    float
    """
    return float((K * L_PE * (A_inlet ** m)) / D)


# ---------------------------------------------------------------------------
# Diagnostic / analysis helpers
# ---------------------------------------------------------------------------

def channel_slope_centerline(
    Z: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Extract slope along the centre-column of a planform grid.

    Parameters
    ----------
    Z : np.ndarray
        ``(Ny, Nx)`` elevation grid.
    Nx, Ny : int
        Grid dimensions.
    dx, dy : float
        Cell spacing (only *dy* is used for the centreline slope).

    Returns
    -------
    np.ndarray
        1-D array of slopes (length *Ny*; last element is NaN).
    """
    ic = (Nx - 1) // 2
    S = np.full(Ny, np.nan, dtype=float)
    for j in range(Ny - 1):
        S[j] = (Z[j, ic] - Z[j + 1, ic]) / dy
    return S


def A_top_for_target_inlet_area(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    A_inlet_target: float,
) -> float:
    """Compute upstream boundary area for a target inlet area.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    Lx, Ly : float
        Domain dimensions.
    A_inlet_target : float
        Desired total inlet area at the top of the centre column.

    Returns
    -------
    float
        ``A_top`` to pass to :func:`compute_area_planform`.
    """
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    ci = (Nx - 1) // 2
    cellA = dx * dy
    left_right = 2.0 * ci * cellA
    return max(A_inlet_target - left_right, 0.0)
