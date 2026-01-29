from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ConvergenceInfo:
    iterations: int
    converged: bool
    last_error: float


class ConformalFactorSolver:
    """Iterative solver for the conformal factor psi = 1 + chi.

    This follows the algorithm you pasted. Where the algorithm says “using FMM”,
    this implementation uses a brute-force particle-particle summation.

    Notes
    -----
    - Points are treated as “particles” located at r_a with associated volumes V_a.
    - The brute-force kernel is O(N^2) per iteration and is intended for small N.
    - A small softening length can be used to avoid singular self-interactions.
    """

    def __init__(
        self,
        r: np.ndarray,
        V: np.ndarray,
        volumes: np.ndarray,
        *,
        M: float,
        softening: float = 0.0,
        exclude_self: bool = True,
        dtype=np.float64,
    ) -> None:
        r = np.asarray(r, dtype=dtype)
        V = np.asarray(V, dtype=dtype)
        volumes = np.asarray(volumes, dtype=dtype)

        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError("r must have shape (N, 3)")
        if V.shape != (r.shape[0],):
            raise ValueError("V must have shape (N,)")
        if volumes.shape != (r.shape[0],):
            raise ValueError("volumes must have shape (N,)")
        if M <= 0:
            raise ValueError("M must be positive")
        if softening < 0:
            raise ValueError("softening must be non-negative")

        self.r = r
        self.V = V
        self.volumes = volumes
        self.M = float(M)
        self.softening = float(softening)
        self.exclude_self = bool(exclude_self)

        self._W = float(np.sum(self.volumes * self.V))

    @property
    def W(self) -> float:
        """Discrete approximation W = ∫ V d^3r."""
        return self._W

    def initial_guess_tilde_chi(self) -> np.ndarray:
        """Default initial guess: \tilde{chi}^{(0)} = (1 + 4 r^2 / M^2)^(-1/2)."""
        r2 = np.sum(self.r * self.r, axis=1)
        return 1.0 / np.sqrt(1.0 + 4.0 * r2 / (self.M * self.M))

    def normalize_chi(self, tilde_chi: np.ndarray) -> Tuple[float, np.ndarray]:
        """Step 6: normalize using the discrete version of the paper's equation."""
        tilde_chi = np.asarray(tilde_chi, dtype=self.r.dtype)
        if tilde_chi.shape != (self.r.shape[0],):
            raise ValueError("tilde_chi must have shape (N,)")

        denom = float(np.sum(self.volumes * self.V * tilde_chi))
        if denom == 0.0:
            raise ZeroDivisionError(
                "Normalization denominator is zero: sum(volumes * V * tilde_chi)"
            )

        C = -(4.0 * np.pi * self.M + self.W) / denom
        chi = C * tilde_chi
        return float(C), chi

    def masses_from_chi(self, chi: np.ndarray) -> np.ndarray:
        """Step 7: m_a = volumes_a * V_a * (1 + chi_a) / (4π)."""
        chi = np.asarray(chi, dtype=self.r.dtype)
        if chi.shape != (self.r.shape[0],):
            raise ValueError("chi must have shape (N,)")
        return (self.volumes * self.V * (1.0 + chi)) / (4.0 * np.pi)

    def _pairwise_kernel(self, targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
        """Return 1/|x_i - y_j| with optional softening."""
        diff = targets[:, None, :] - sources[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        if self.softening > 0.0:
            d2 = d2 + self.softening * self.softening
        return 1.0 / np.sqrt(d2)

    def chi_from_masses(self, m: np.ndarray) -> np.ndarray:
        """Brute-force version of step 9.

        Uses the same kernel implied by step 12:
            psi(r) = 1 - (1/4π) * Σ_a m_a / |r - r_a|
        hence
            chi(r) = psi - 1 = - (1/4π) * Σ_a m_a / |r - r_a|.

        For points exactly coincident with a source, this implementation either
        excludes self-interactions (default) or uses a softening length.
        """
        m = np.asarray(m, dtype=self.r.dtype)
        if m.shape != (self.r.shape[0],):
            raise ValueError("m must have shape (N,)")

        inv_r = self._pairwise_kernel(self.r, self.r)
        if self.exclude_self and self.softening == 0.0:
            np.fill_diagonal(inv_r, 0.0)

        potential = inv_r @ m
        return -(1.0 / (4.0 * np.pi)) * potential

    def solve(
        self,
        *,
        tol: float = 1e-10,
        max_iter: int = 200,
        error_norm: Literal["linf", "l2"] = "linf",
        verbose: bool = False,
        initial_tilde_chi: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ConvergenceInfo]:
        """Run steps 4-11 on the interpolation points.

        Returns
        -------
        psi : ndarray, shape (N,)
            Conformal factor at the interpolation points.
        info : ConvergenceInfo
            Iteration diagnostics.
        """
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        if initial_tilde_chi is None:
            tilde_chi = self.initial_guess_tilde_chi()
        else:
            tilde_chi = np.asarray(initial_tilde_chi, dtype=self.r.dtype)

        last_err = np.inf
        for n in range(max_iter):
            _, chi = self.normalize_chi(tilde_chi)
            m = self.masses_from_chi(chi)

            tilde_next = self.chi_from_masses(m)

            delta = tilde_next - tilde_chi
            if error_norm == "linf":
                err = float(np.max(np.abs(delta)))
            elif error_norm == "l2":
                err = float(np.linalg.norm(delta) / np.sqrt(delta.size))
            else:
                raise ValueError("error_norm must be 'linf' or 'l2'")

            if verbose:
                print(f"iter {n:4d}  err={err:.3e}")

            tilde_chi = tilde_next
            last_err = err
            if err < tol:
                _, chi = self.normalize_chi(tilde_chi)
                psi = 1.0 + chi
                return psi, ConvergenceInfo(iterations=n + 1, converged=True, last_error=err)

        # not converged
        _, chi = self.normalize_chi(tilde_chi)
        psi = 1.0 + chi
        return psi, ConvergenceInfo(iterations=max_iter, converged=False, last_error=last_err)

    def psi_at_points(self, r_eval: np.ndarray, *, chi: np.ndarray) -> np.ndarray:
        """Step 12: evaluate psi(r) at arbitrary points via brute-force summation."""
        r_eval = np.asarray(r_eval, dtype=self.r.dtype)
        if r_eval.ndim != 2 or r_eval.shape[1] != 3:
            raise ValueError("r_eval must have shape (K, 3)")

        m = self.masses_from_chi(chi)
        inv_r = self._pairwise_kernel(r_eval, self.r)

        potential = inv_r @ m
        return 1.0 - (1.0 / (4.0 * np.pi)) * potential
