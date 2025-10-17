import math
from dataclasses import dataclass
from typing import Sequence, Tuple
from scipy.constants import Avogadro as N_A, Boltzmann as kB

@dataclass
class SHLMixture:
    """
    Spinodal Homogeneous Liquid (SHL) nucleation for mixtures (bubble/cavitation).

    Required inputs:
      - Jcrit : critical nucleation rate [1/m^3·s]
      - Psat  : saturation pressure [Pa]
      - T     : temperature [K]
      - gamma : surface tension [N/m]
      - rho_L : liquid mass density [kg/m^3]
      - M_list: molar masses [kg/mol]
      - x_list: liquid mole fractions (sum to 1)
    """
    Jcrit: float
    Psat: float
    T: float
    gamma: float
    rho_L: float
    M_list: Sequence[float]
    x_list: Sequence[float]

    # ---------- internals / helpers ----------
    def _check_inputs(self) -> None:
        if len(self.M_list) != len(self.x_list):
            raise ValueError("M_list and x_list must have the same length.")
        s = sum(self.x_list)
        if abs(s - 1.0) > 1e-12:
            raise ValueError(f"x_list must sum to 1 (got {s}).")
        if self.Jcrit <= 0:
            raise ValueError("Jcrit must be > 0.")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if self.T <= 0:
            raise ValueError("T must be > 0 K.")

    def _rho_nL(self) -> float:
        """Liquid number density [1/m^3]."""
        M_mix = sum(xi * Mi for xi, Mi in zip(self.x_list, self.M_list))  # kg/mol
        return (self.rho_L / M_mix) * N_A

    # ---------- public API ----------
    def K(self) -> float:
        """Prefactor K for SHL [1/m^3·s]."""
        self._check_inputs()
        rho_nL = self._rho_nL()
        sum_term = sum(xi * math.sqrt(N_A / Mi) for xi, Mi in zip(self.x_list, self.M_list))
        return rho_nL * math.sqrt(2.0 * self.gamma / math.pi) * sum_term

    def compute(self) -> Tuple[float, float, float]:
        """
        Returns (DeltaP [Pa], Pc [Pa], rc [m]).
        """
        K_val = self.K()
        if K_val <= self.Jcrit:
            raise ValueError(f"K must be > Jcrit for CNT (got K={K_val:.3e}, J={self.Jcrit:.3e}).")

        A = 16.0 * math.pi * self.gamma**3 / (3.0 * kB * self.T)
        DeltaP = math.sqrt(A / math.log(K_val / self.Jcrit))
        Pc = self.Psat - DeltaP
        rc = 2.0 * self.gamma / DeltaP
        return DeltaP, Pc, rc

@dataclass
class SCLMixture:
    """
    Spinodal Condensed Liquid (SCL) nucleation for mixtures (droplet/condensation).

    Required inputs:
      - Jcrit : critical nucleation rate [1/m^3·s]
      - Psat  : saturation pressure [Pa]
      - T     : temperature [K]
      - gamma : surface tension [N/m]
      - rho_L : liquid mass density [kg/m^3]
      - rho_g : vapor mass density [kg/m^3]
      - M_list: component molar masses [kg/mol]
      - x_list: liquid mole fractions (sum=1)
      - y_list: vapor mole fractions (sum=1)
    """
    Jcrit: float
    Psat: float
    T: float
    gamma: float
    rho_L: float
    rho_g: float
    M_list: Sequence[float]
    x_list: Sequence[float]
    y_list: Sequence[float]

    # ---------- internals ----------
    def _check_inputs(self) -> None:
        if len(self.M_list) != len(self.x_list) or len(self.M_list) != len(self.y_list):
            raise ValueError("M_list, x_list, and y_list must have the same length.")
        if abs(sum(self.x_list) - 1.0) > 1e-12:
            raise ValueError("x_list must sum to 1.")
        if abs(sum(self.y_list) - 1.0) > 1e-12:
            raise ValueError("y_list must sum to 1.")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if self.T <= 0:
            raise ValueError("T must be > 0 K.")
        if self.Jcrit <= 0:
            raise ValueError("Jcrit must be > 0.")

    def _rho_number(self, rho_mass: float, comp_list: Sequence[float], mol_list: Sequence[float]) -> float:
        """Converts mass density [kg/m³] to number density [1/m³]."""
        M_mix = sum(ci * Mi for ci, Mi in zip(comp_list, mol_list))
        return (rho_mass / M_mix) * N_A

    def _m_eff(self) -> float:
        """Effective molecular mass (per molecule) for vapor mixture [kg]."""
        inv_sqrt = sum(y_i * math.sqrt(N_A / M_i) for y_i, M_i in zip(self.y_list, self.M_list))
        return 1.0 / (inv_sqrt**2)

    # ---------- prefactor ----------
    def K(self) -> float:
        """Prefactor K for SCL [1/m^3·s]."""
        self._check_inputs()
        rho_nL = self._rho_number(self.rho_L, self.x_list, self.M_list)
        rho_nV = self._rho_number(self.rho_g, self.y_list, self.M_list)
        m_eff = self._m_eff()
        return (rho_nV**2 / rho_nL) * math.sqrt(2.0 * self.gamma / (math.pi * m_eff))

    # ---------- main computation ----------
    def compute(self) -> Tuple[float, float, float]:
        """
        Returns (Pc [Pa], rc [m], lnS) for SCL.
        """
        K_val = self.K()
        rho_nL = self._rho_number(self.rho_L, self.x_list, self.M_list)
        A_over_kBT = 16.0 * math.pi * self.gamma**3 / (3.0 * (rho_nL**2) * (kB * self.T)**3)
        lnS = math.sqrt(A_over_kBT / math.log(K_val / self.Jcrit))
        Pc = self.Psat * math.exp(lnS)
        rc = 2.0 * self.gamma / (rho_nL * kB * self.T * lnS)
        return Pc, rc, lnS
