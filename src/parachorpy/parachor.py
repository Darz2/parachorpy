#!/usr/bin/env python

# Created by Darshan on 2025-07-15

import json,os,re,math,numpy as np, collections.abc
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from pathlib import Path
from scipy.constants import Avogadro, Boltzmann
from scipy.optimize import brentq
from si_units import PASCAL, BAR

class InterfacialTension:
    def __init__(self, json_files):
            """
            Load one or more JSON files from the database directory.

            Parameters
            ----------
            json_files : str | list[str]
                Filename(s) or full path(s) to JSON files.
            """
            if isinstance(json_files, str):
                json_files = [json_files]

            # Default database directory: repo_root/database
            base_dir = Path(__file__).resolve().parents[2]  # up from src/parachorpy/
            db_dir = base_dir / "database"

            self.data = []

            for jf in json_files:
                path = Path(jf)

                # If not an absolute/relative path, assume it's in database/
                if not path.is_absolute() and not path.exists():
                    path = db_dir / jf

                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")

                with open(path, "r", encoding="utf-8") as f:
                    self.data.extend(json.load(f))

    def ctREFPROP_init(self, rp_path, gerg_enable=0):
        """
        Initialize REFPROP with the given path.

        Parameters:
        - rp_path : str
            Path to REFPROP installation/build folder
        - gerg_enable : int, optional
            1 to enable GERG-2008 model, 0 to disable (default: 0)
        """
        os.environ['RPPREFIX'] = os.path.expanduser(rp_path)

        RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
        RP.SETPATHdll(os.environ['RPPREFIX'])
        print("REFPROP version =", RP.RPVersion())

        # TO SET GERG-2008 MODEL if requested
        if gerg_enable == 1:
            RP.FLAGSdll('GERG', 1)
            print("GERG-2008 model = Enabled")
            self.RP_GERG = RP
        else:
            RP.FLAGSdll('GERG', 0)
            print("GERG-2008 model = Disabled")
        
        self.RP = RP

        # TO SET MOLAR UNITS
        self.MOLAR_BASE = RP.GETENUMdll(0, "MOLAR BASE SI").iEnum
        
        # TO SET SI UNITS
        self.SI_BASE = RP.GETENUMdll(0, "SI").iEnum

        # # TEST
        # CO2_SI      = self.RP.REFPROPdll("CO2", "TQ", "D;P", self.SI_BASE, 0, 0, 220, 0, [1.0])
        # CO2_MOLAR   = self.RP.REFPROPdll("CO2", "TQ", "D;P", self.MOLAR_BASE, 0, 0, 220, 0, [1.0])
        # print(f"Liquid density of CO2 at 220K: {CO2_SI.Output[0]:.2f} kg/m3")
        # MM_CO2      = self.RP.REFPROPdll("CO2", "TQ", "M", self.SI_BASE, 0, 0, 220, 0, [1.0])
        # print(f"Molar mass of CO2: {MM_CO2.Output[0]:.4f} kg/kmol")
        # print(f"Liquid density of CO2 at 220K: {CO2_MOLAR.Output[0]/1e3:.2f} Kmol/m3")

    def _normalize_name(self, name: str) -> str:
        # lower, remove non-alphanumerics
        return re.sub(r'[^a-z0-9]', '', name.lower())

    # Use normalized keys here
    _FLUID_ALIASES = {
        'co2': 'carbondioxide',
        'r744': 'carbondioxide',
        'carbondioxide': 'carbondioxide',
    }

    def get_fluid_params(self, fluid_name):
        """
        Extract Tc, s, n for a given fluid. Accepts aliases like CO2, R744, R-744, carbon dioxide.
        """
        q = self._normalize_name(fluid_name)
        target = self._FLUID_ALIASES.get(q, q)

        for entry in self.data:
            entry_name = self._normalize_name(entry.get("Fluid", ""))
            if entry_name == target:
                Tc = entry.get("Tc", None)
                return Tc, entry["s"], entry["n"]

        raise ValueError(f"Fluid '{fluid_name}' not found (normalized as '{q}', alias -> '{target}').")

    def compute_gamma_pure(self, T, fluid_name, key, Tc=None):
        """
        Compute Interfacial tension (IFT) for a fluid at temperature T.

        Parameters:
        - T : float
        - fluid_name : str
        - Tc : float (optional) — use only if Tc is not in the JSON

        Returns:
        - gamma : float (mN/m)
        """
        Tc_json, s, n = self.get_fluid_params(fluid_name)
        Tc_final = Tc_json if Tc_json is not None else Tc

        if Tc_final is None:
            raise ValueError(f"No Tc provided in JSON or as input for {fluid_name}")

        if len(s) != len(n):
            raise ValueError("Length of s and n must be equal")

        if key == 'Parachor':
            if T <= Tc_final:
                gamma_T = sum(si * (1 - T / Tc_final)**ni for si, ni in zip(s, n))

            if T > Tc_final:
                print(f"Given temperature is greater than Tc ({Tc_final}) for {fluid_name}")
                print(f"Computing interfacial tension at T = 0.9*{Tc_final} for Parachor method")
                T = 0.9 * Tc_final
                gamma_T = sum(si * (1 - T / Tc_final)**ni for si, ni in zip(s, n))
        
        elif key == "WSD":
            if T <= Tc_final:
                gamma_T = sum(si * (1 - T / Tc_final)**ni for si, ni in zip(s, n))
            
            if T > Tc_final:
                gamma_T = 0    
        
        else:
            print("The key should be either Parachor or WSD")
            gamma_T = np.NaN 

        return gamma_T * 1e3, T  # Convert to mN/m

    def parachor_number(self,rho_l, rho_v, gamma):
        """
        Calculate the parachor number.

        Parameters:
        - rho_l : float — Liquid density [kg/m³]
        - rho_v : float — Vapor density [kg/m³]
        - gamma : float — Interfacial tension [mN/m]

        Returns:
        - P : float — parachor number [kg/m³]ⁿ·mN/m

        Constant:
        - n : float — exponent in the parachor equation (default: 3.87); REFPROP v10 (Lemmon et al, 2018)
        """
        n = 3.87
        if rho_l <= 0 or rho_v <= 0 or gamma <= 0:
            raise ValueError("All inputs must be positive")

        delta_rho = rho_l - rho_v               # [mol/cm³]
        gamma_SI  = gamma                        # in  mN/m
        parachor  = gamma_SI**(1/n) / delta_rho
        
        return parachor

    def compute_gamma_Parachor(self, x, y, rho_l, rho_v, parachor_numbers, kij=0.0):
        """
        Compute Interfacial tension of a mixture using the parachor method.

        Parameters:
        - x, y : mole fractions (lists) for liquid and vapor phases
        - rho_l, rho_v : densities [mol/cm^3]
        - parachor_numbers : list of component parachors (P_i)
        - kij : scalar or matrix of binary interaction parameters (default 0)

        Returns:
        - gamma_mix : Interfacial tension [mN/m]
        """
        if not (len(x) == len(y) == len(parachor_numbers)):
            raise ValueError("Mismatched input lengths")

        N = len(x)
        n = 3.87

        # allow kij to be scalar OR NxN matrix
        def kij_ij(i, j):
            if hasattr(kij, "__len__"):
                return kij[i][j]
            return kij

        def P_ij(i, j):
            if i == j:
                return parachor_numbers[i]
            return (1 - kij_ij(i, j)) * 0.5 * (parachor_numbers[i] + parachor_numbers[j])

        P_l = sum(x[i] * x[j] * P_ij(i, j) for i in range(N) for j in range(N))
        P_v = sum(y[i] * y[j] * P_ij(i, j) for i in range(N) for j in range(N))

        gamma_mix_Parachor = (rho_l * P_l - rho_v * P_v) ** n
        
        return gamma_mix_Parachor
    
    # Created by Darshan on 2025-10-01
    def compute_gamma_WSD(self, T, comp, x, y, rho_l, rho_v, phi=1.0):
        """
        Compute Interfacial tension of a mixture using the Winterfeld-Scriven-Davis method.
        DOI:10.1002/aic.690240610
        
        Matrix form: gamma = a^T G a, where
        a_i = (x_i*rho_l - y_i*rho_v) / (n^0_{i,l} - n^0_{i,v})
        G_ii = gamma_i^0
        G_ij = Phi_ij * sqrt(gamma_i^0 * gamma_j^0),  i != j
        
        If T > Tc of component i, that component is omitted (a_i=0 and G entries
        in row/col i are zero), so only components with T <= Tc contribute.
        
        Parameters
        ----------
        T : float
            Temperature [K]
        comp : list[str]
            Component IDs (REFPROP-compatible), length N
        x, y : list[float]
            Mole fractions for liquid and vapor, length N
        rho_l, rho_v : float
            Mixture molar densities [mol/cm^3]
        phi : float or 2D list/array
            Mixing parameter(s). If scalar, Phi_ij = phi for all i != j.
            By default it is 1

        Returns:
        - gamma_mix : Interfacial tension [mN/m]
        """
        
        if not (len(comp) == len(x) == len(y)):
            raise ValueError("Mismatched input lengths: comp, x, and y must have same length.")
        N = len(comp)

        def phi_ij(i, j):
            if hasattr(phi, "__len__"):
                return phi[i][j]
            return phi

        # --- Step 0: get Tc_i from the JSON so we can decide which components are active ---
        Tc_list = []
        for ci in comp:
            Tc_i, _s, _n = self.get_fluid_params(ci)
            if Tc_i is None:
                raise ValueError(f"No Tc found for {ci} in JSON.")
            Tc_list.append(Tc_i)

        active = [T <= Tc_i for Tc_i in Tc_list]
        
        if not any(active):
            return 0.0  # nothing contributes above all critical temps
    
        # --- Step 1: Pure-component data at T (saturated liquid/vapor) ---
        
        # We need: n0_i = (n_{i,l}^0 - n_{i,v}^0) [mol/cm^3], and gamma_i^0 [mN/m]
        n0      = np.zeros(N)        # pure liquid–vapor molar-density difference
        gamma0  = np.zeros(N)        # pure interfacial tension
        
        for i, ci in enumerate(comp):
            
            if not active[i]:
                n0[i] = 1.0      # dummy to avoid division by zero; a_i will be forced to 0 below
                gamma0[i] = 0.0
                continue

            MM = self.RP.REFPROPdll(ci, "TQ", "M", self.SI_BASE, 0, 0, T, 0, [1.0]).Output[0]  # Molar mass [g/mol]

            # Densities from REFPROP in kg/m^3 -> convert to mol/cm^3:
            # (kg/m^3) / (MM[g/mol]*1e-3 kg/g) -> mol/m^3; then /1e6 -> mol/cm^3
            rhoL_mol_cm3 = self.RP.REFPROPdll(ci, "TQ", "D", self.SI_BASE, 0, 0, T, 0, [1.0]).Output[0] / (MM * 1e3)
            rhoV_mol_cm3 = self.RP.REFPROPdll(ci, "TQ", "D", self.SI_BASE, 0, 0, T, 1, [1.0]).Output[0] / (MM * 1e3)

            n0[i] = rhoL_mol_cm3 - rhoV_mol_cm3  # denominator for a_i

            if abs(n0[i]) < 1e-30:
                # This can happen extremely close to Tc; treat as inactive.
                active[i] = False
                n0[i] = 1.0
                gamma0[i] = 0.0
                continue

            # Pure interfacial tension [mN/m]
            gamma0[i], _ = self.compute_gamma_pure(T, ci, key='WSD', Tc=None)
                    
        # --- Step 2: Mixture coefficients a_i (from mixture and pure n's) ---
        # a_i = (x_i*rho_l - y_i*rho_v) / (n^0_{i,l} - n^0_{i,v})
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        a = (x * rho_l - y * rho_v) / n0  # shape (N,)
        
        # force inactive ones to 0 so they don't contribute anywhere
        for i in range(N):
            if not active[i]:
                a[i] = 0.0
        
        # --- Step 3: Build symmetric interaction matrix G ---
        # G_ii = gamma_i^0; G_ij = Phi_ij * sqrt(gamma_i^0 * gamma_j^0), i != j
        
        G = np.zeros((N, N), dtype=float)
        for i in range(N):
            if active[i]:
                G[i, i] = gamma0[i]
            for j in range(i + 1, N):
                if active[i] and active[j]:
                    gij = phi_ij(i, j) * math.sqrt(gamma0[i] * gamma0[j])
                    G[i, j] = G[j, i] = gij
                # else keep zeros if any partner is inactive

        # --- Step 4: Quadratic form ---
        gamma_mix_WSD = float(a @ G @ a)  # [mN/m]
        
        
        # ---- Step 4: Supercritical correction (Roar correction) ----
        if not all(active):
            # sum of liquid mole fractions of inactive species
            x_inactive_sum = float(np.sum(x[[i for i in range(N) if not active[i]]]))
            mixcorr = max(0.0, 1.0 - x_inactive_sum)   # guard against tiny negatives
            gamma_mix_WSD *= mixcorr
        
        return gamma_mix_WSD
        
    def gamma_TP(self, mixture: str, z: list[float], T: float,pressures_bar,
                        kij: float = 0.0, phi_ij=1.0):
        """
        Compute the interfacial tension (IFT) of a binary mixture along its
        phase envelope (TP diagram) at fixed temperature.
        
        This method uses both the Parachor and Winterfeld-Scriven-Davis (WSD)
        models to evaluate IFT at a series of equilibrium state points
        defined by pressure, composition, and densities.

        Parameters
        ----------
        mixture : str
            REFPROP mixture string, e.g. "CO2;Methane"
        z : list[float]
            Overall composition (same order as `mixture` components)
        T : float
            Temperature [K]
        pressures_bar : float | iterable of float
            Pressure(s) in bar
        kij : float
            Binary interaction parameter for parachor mixing rule (scalar)
        phi_ij : float
            Mixing parameter for WSD model (scalar)    

        Returns
        -------
        pressures_bar           : list[float]
        results_gamma_Parachor  : list[float]
        results_gamma_WSD       : list[float]
        """

        # Allow single float or iterable
        if isinstance(pressures_bar, collections.abc.Iterable) and not isinstance(pressures_bar, (str, bytes)):
            pressures_bar = list(pressures_bar)
        else:
            pressures_bar = [pressures_bar]
        
        components      = mixture.split(";")
        ncomp           = len(components)
        pressures_bar   = list(pressures_bar)

        # --- per-component molar masses & parachor numbers at Teff (from pure gamma)
        MOLAR_MASSES        = []
        parachor_numbers    = []

        for comp in components:
            # Molar mass (kg/kmol == g/mol numerically)
            MM = self.RP.REFPROPdll(comp, "TQ", "M", self.SI_BASE, 0, 0, T, 0, [1.0]).Output[0]
            MOLAR_MASSES.append(MM)

            # Pure gamma (may switch to Teff=0.9Tc if T>Tc)
            gamma_val_mNpm, Teff = self.compute_gamma_pure(T, comp, key='Parachor', Tc=None)

            liq = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 0, [1.0])
            vap = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 1, [1.0])

            rhoL_kg_m3 = liq.Output[0]
            rhoV_kg_m3 = vap.Output[0]

            # Convert densities to mol/cm^3 and compute parachor number
            P_i = self.parachor_number(
                rhoL_kg_m3 / (MM * 1e3),
                rhoV_kg_m3 / (MM * 1e3),
                gamma_val_mNpm
            )
            parachor_numbers.append(P_i)

        results_gamma_Parachor  = []
        results_gamma_WSD       = []
        
        results_MM_L            = []
        results_MM_V            = []
        results_rhoL_kg_m3      = []
        results_rhoV_kg_m3      = []   
        results_xL              = []
        results_yV              = []     
        
        for Pbar in pressures_bar:
            
            # # Debug prints 
            # print(f"\nCalculating mixture gamma at T={T} K, P={Pbar} bar for {mixture}")

            mix = self.RP.REFPROPdll(mixture, "TP", "D;Dliq;Dvap", self.SI_BASE, 0, 1, T, Pbar/10.0, z)
            
            xL  = np.array(mix.x[:ncomp])
            yV  = np.array(mix.y[:ncomp])

            results_xL.append(xL)
            results_yV.append(yV)
            
            rhoL_kg_m3 = mix.Output[1]
            rhoV_kg_m3 = mix.Output[2]
            
            results_rhoL_kg_m3.append(rhoL_kg_m3)
            results_rhoV_kg_m3.append(rhoV_kg_m3)
            
            # phase molar masses (kg/kmol == g/mol numerically)
            MM_L = float(np.sum(xL * MOLAR_MASSES))
            MM_V = float(np.sum(yV * MOLAR_MASSES))
            
            results_MM_L.append(MM_L)
            results_MM_V.append(MM_V)

            # kg/m3 -> (kg/kmol) -> kmol/m3 -> mol/cm3
            rhoL_mol_cm3 = (rhoL_kg_m3 / MM_L) * 1e-3
            rhoV_mol_cm3 = (rhoV_kg_m3 / MM_V) * 1e-3

            gamma_mix_Parachor = self.compute_gamma_Parachor(
                                x=xL.tolist(),y=yV.tolist(),
                                rho_l=rhoL_mol_cm3,rho_v=rhoV_mol_cm3,
                                parachor_numbers=parachor_numbers,kij=kij)
            
            gamma_mix_WSD = self.compute_gamma_WSD(
                            T=T, comp = components,x=xL.tolist(),
                            y=yV.tolist(),rho_l=rhoL_mol_cm3,
                            rho_v=rhoV_mol_cm3,phi=phi_ij)
            
            results_gamma_Parachor.append(gamma_mix_Parachor)
            results_gamma_WSD.append(gamma_mix_WSD)
            
        return pressures_bar, results_gamma_Parachor, results_gamma_WSD, \
            results_xL, results_yV, results_rhoL_kg_m3, results_rhoV_kg_m3, results_MM_L, results_MM_V
    
    def apply_mask_and_sort(self, P, arrays):
        """
        Apply finite mask on P, drop NaNs, and sort all arrays by P.

        Parameters
        ----------
        P : np.ndarray
            Pressures [bar].
        arrays : dict[str, np.ndarray]
            Arrays with the same length as P.

        Returns
        -------
        P_sorted : np.ndarray
        arrays_sorted : dict[str, np.ndarray]
        """
        mask = np.isfinite(P)
        P = P[mask]
        arrays = {k: v[mask] for k, v in arrays.items()}

        idx = np.argsort(P)
        P = P[idx]
        arrays = {k: v[idx] for k, v in arrays.items()}
        
        return P, arrays
    
    def decreasing_mask(self, P, gamma, rel_drop=0.20, abs_drop=0.2, near_zero=1e-4):
        """
        Remove isolated downward spikes in gamma(P) (V-shaped sudden drops).

        Parameters
        ----------
        P : array-like [bar]
        gamma : array-like [mN/m]
        rel_drop : float
            Minimum relative drop vs. neighbors to flag a spike (e.g. 0.20 = 20%).
        abs_drop : float
            Minimum absolute drop [mN/m] vs. neighbors to flag a spike.
        near_zero : float
            If gamma[i] < near_zero but neighbors are much larger, treat as spike.

        Returns
        -------
        keep : boolean array, same shape as P
            True for points to keep, False for spikes to remove.
        idx_sort : argsort indices used to sort by pressure
        """
        P = np.asarray(P, float)
        g = np.asarray(gamma, float)
        n = len(g)
        idx_sort = np.argsort(P)
        g_sorted = g[idx_sort]

        keep_sorted = np.ones(n, dtype=bool)

        for i in range(1, n-1):
            gi_prev = g_sorted[i-1]
            gi      = g_sorted[i]
            gi_next = g_sorted[i+1]

            # Must have finite triplet
            if not (np.isfinite(gi_prev) and np.isfinite(gi) and np.isfinite(gi_next)):
                continue

            # Typical scale to compare against (use the larger neighbor)
            scale = max(abs(gi_prev), abs(gi_next), 1.0)  # avoid scale~0

            # Conditions for a "V-shaped" downward spike at i:
            #  1) current point far below BOTH neighbors (abs AND relative)
            big_abs_drop = (min(gi_prev, gi_next) - gi) > abs_drop
            big_rel_drop = (min(gi_prev, gi_next) - gi) > rel_drop * scale

            #  2) and the series *rebounds* after the spike
            rebound = (gi_next - gi) > max(abs_drop, rel_drop * scale)

            #  3) also treat near-zero blips as spikes if neighbors are not near-zero
            near_zero_blip = (gi < near_zero) and (max(gi_prev, gi_next) > 5*near_zero)

            if (big_abs_drop and big_rel_drop and rebound) or near_zero_blip:
                keep_sorted[i] = False

        # Map mask back to original order
        keep = np.zeros_like(keep_sorted, dtype=bool)
        keep[idx_sort] = keep_sorted
        return keep, idx_sort

    def RP_PXY(self, mixture: str, T: float, npts: int,
                    x1_grid=None, y1_grid=None, clip=1e-4, 
                    verbose: bool = False, return_densities: bool =True):
        """
        Pxy at fixed T for a *binary* mixture using REFPROP 'TQ' mode.

        Bubble line (Q=0): input x (liquid comp) -> get P and coexisting y.
        Dew    line (Q=1): input y (vapor  comp) -> get P and coexisting x.

        Returns
        -------
        out: dict
        'bubble': {'x','y','P_bar','M_liq','M_vap', ['rhoL','rhoV','cL_mol_cm3','cV_mol_cm3' if return_densities]}
        'dew'   : {'x','y','P_bar','M_liq','M_vap', ['rhoL','rhoV','cL_mol_cm3','cV_mol_cm3' if return_densities]}
        Units:
        rho* in kg/m^3 (SI), M_* in kg/mol, c*_mol_cm3 in mol/cm^3.
        """
        comps = mixture.split(';')
        if len(comps) != 2:
            raise ValueError("REFPROP_PXY supports binary mixtures only, e.g. 'CO2;Methane'.")

        # Build grids; avoid exact 0 and 1
        if x1_grid is None:
            x1_grid = np.linspace(clip, 1.0 - clip, npts)
        else:
            x1_grid = np.clip(np.asarray(list(x1_grid), float), clip, 1.0 - clip)

        if y1_grid is None:
            y1_grid = np.linspace(clip, 1.0 - clip, npts)
        else:
            y1_grid = np.clip(np.asarray(list(y1_grid), float), clip, 1.0 - clip)

        Nb, Nd = len(x1_grid), len(y1_grid)

        # Allocate (bubble)
        bub_x            = np.full((Nb, 2), np.nan)
        bub_y            = np.full((Nb, 2), np.nan)
        bub_P_bar        = np.full(Nb, np.nan)
        bub_Dliq_kg_m3   = np.full(Nb, np.nan)
        bub_Dvap_kg_m3   = np.full(Nb, np.nan)
        bub_ML_kg_mol    = np.full(Nb, np.nan)
        bub_MV_kg_mol    = np.full(Nb, np.nan)
        bub_cL_mol_cm3   = np.full(Nb, np.nan)
        bub_cV_mol_cm3   = np.full(Nb, np.nan)

        # Allocate (dew)
        dew_x            = np.full((Nd, 2), np.nan)
        dew_y            = np.full((Nd, 2), np.nan)
        dew_P_bar        = np.full(Nd, np.nan)
        dew_Dliq_kg_m3   = np.full(Nd, np.nan)
        dew_Dvap_kg_m3   = np.full(Nd, np.nan)
        dew_ML_kg_mol    = np.full(Nd, np.nan)
        dew_MV_kg_mol    = np.full(Nd, np.nan)
        dew_cL_mol_cm3   = np.full(Nd, np.nan)
        dew_cV_mol_cm3   = np.full(Nd, np.nan)

        # Helper to check REFPROP success robustly
        def ok_res(res):
            ierr = getattr(res, "ierr", 0)
            herr = getattr(res, "herr", "")
            out0 = res.Output[0] if (hasattr(res, "Output") and len(res.Output) > 0) else -1e20
            return (ierr == 0) and (herr.strip() == "") and (out0 > -1e10)

        # -------- Bubble line: TQ, Q=0, z=x --------
        for k, x1 in enumerate(x1_grid):
            x = [float(x1), float(1.0 - x1)] 
            res = self.RP.REFPROPdll(mixture, "TQ", "P;Dliq;Dvap;M", self.SI_BASE, 0, 0, T, 0.0, x)
            if ok_res(res):
                bub_P_bar[k]      = res.Output[0] * 10.0    # MPa -> bar
                bub_Dliq_kg_m3[k] = res.Output[1]           # kg/m^3 (liq)
                bub_Dvap_kg_m3[k] = res.Output[2]           # kg/m^3 (vap)
                M_g_per_mol       = res.Output[3]           # g/mol (liq, Q=0)
                bub_ML_kg_mol[k]  = M_g_per_mol / 1000.0    # kg/mol
                bub_x[k, :]       = x
                bub_y[k, 0:2]     = res.y[0:2]

                # Coexisting vapor at same T, Q=1 with y
                try:
                    res_v = self.RP.REFPROPdll(mixture, "TQ", "M", self.SI_BASE, 0, 0, T, 1.0, bub_y[k, :].tolist())
                    if ok_res(res_v):
                        bub_MV_kg_mol[k] = res_v.Output[0] / 1000.0
                except Exception:
                    if verbose:
                        print(f"[BUB extra M] failed at k={k}")

                # Molar concentrations (mol/cm^3)
                if return_densities:
                    if np.isfinite(bub_ML_kg_mol[k]) and bub_ML_kg_mol[k] > 0:
                        bub_cL_mol_cm3[k] = (bub_Dliq_kg_m3[k] / bub_ML_kg_mol[k]) / 1e6
                    if np.isfinite(bub_MV_kg_mol[k]) and bub_MV_kg_mol[k] > 0:
                        bub_cV_mol_cm3[k] = (bub_Dvap_kg_m3[k] / bub_MV_kg_mol[k]) / 1e6
            elif verbose:
                print(f"[BUB] T={T} K, x1={x1:.4f} failed: ierr={getattr(res,'ierr',None)} herr='{getattr(res,'herr','')}'")

        # -------- Dew line: TQ, Q=1, z=y --------
        for k, y1 in enumerate(y1_grid):
            y = [float(y1), float(1.0 - y1)]
            res = self.RP.REFPROPdll(mixture, "TQ", "P;Dliq;Dvap;M", self.SI_BASE, 0, 0, T, 1.0, y)
            if ok_res(res):
                dew_P_bar[k]      = res.Output[0] * 10.0
                dew_Dliq_kg_m3[k] = res.Output[1]
                dew_Dvap_kg_m3[k] = res.Output[2]
                M_g_per_mol       = res.Output[3]         # g/mol (vap, Q=1)
                dew_MV_kg_mol[k]  = M_g_per_mol / 1000.0  # kg/mol
                dew_y[k, :]       = y
                dew_x[k, 0:2]     = res.x[0:2]

                # Coexisting liquid at same T, Q=0 with x
                try:
                    res_l = self.RP.REFPROPdll(mixture, "TQ", "M", self.SI_BASE, 0, 0, T, 0.0, dew_x[k, :].tolist())
                    if ok_res(res_l):
                        dew_ML_kg_mol[k] = res_l.Output[0] / 1000.0
                except Exception:
                    if verbose:
                        print(f"[DEW extra M] failed at k={k}")

                # Molar concentrations (mol/cm^3)
                if return_densities:
                    if np.isfinite(dew_ML_kg_mol[k]) and dew_ML_kg_mol[k] > 0:
                        dew_cL_mol_cm3[k] = (dew_Dliq_kg_m3[k] / dew_ML_kg_mol[k]) / 1e6
                    if np.isfinite(dew_MV_kg_mol[k]) and dew_MV_kg_mol[k] > 0:
                        dew_cV_mol_cm3[k] = (dew_Dvap_kg_m3[k] / dew_MV_kg_mol[k]) / 1e6
            elif verbose:
                print(f"[DEW] T={T} K, y1={y1:.4f} failed: ierr={getattr(res,'ierr',None)} herr='{getattr(res,'herr','')}'")

        # --- Drop NaNs and sort consistently ---
        bub_arrays = {
            "x": bub_x, "y": bub_y,
            "M_liq": bub_ML_kg_mol, "M_vap": bub_MV_kg_mol,
        }
        if return_densities:
            bub_arrays.update({
                "rhoL_kg_m3": bub_Dliq_kg_m3, "rhoV_kg_m3": bub_Dvap_kg_m3,
                "rhoL_mol_cm3": bub_cL_mol_cm3, "rhoV_mol_cm3": bub_cV_mol_cm3,
            })
        bub_P_bar, bub_arrays = self.apply_mask_and_sort(bub_P_bar, bub_arrays)

        dew_arrays = {
            "x": dew_x, "y": dew_y,
            "M_liq": dew_ML_kg_mol, "M_vap": dew_MV_kg_mol,
        }
        if return_densities:
            dew_arrays.update({
                "rhoL_kg_m3": dew_Dliq_kg_m3, "rhoV_kg_m3": dew_Dvap_kg_m3,
                "rhoL_mol_cm3": dew_cL_mol_cm3, "rhoV_mol_cm3": dew_cV_mol_cm3,
            })
        dew_P_bar, dew_arrays = self.apply_mask_and_sort(dew_P_bar, dew_arrays)

        # Package output
        pxy_out = {
            "bubble": {"P_bar": bub_P_bar, **bub_arrays},
            "dew": {"P_bar": dew_P_bar, **dew_arrays},
        }
        
        return pxy_out

    def gamma_pxy(self, mixture: str, T: float, P: list[float], x: list[float], y: list[float],
                 rhoL: list[float], rhoV: list[float],kij: float = 0.0, phi_ij=1.0, verbose: bool = False,
                 enforce_monotone: bool = True, rel_tol=0.05, abs_tol=1e-6):
        """
        Compute the interfacial tension (IFT) of a binary mixture along its
        phase envelope (Pxy diagram) at fixed temperature.

        This method uses both the Parachor and Winterfeld-Scriven-Davis (WSD)
        models to evaluate IFT at a series of equilibrium state points
        defined by pressure, composition, and densities.

        Parameters
        ----------
        mixture : str
            REFPROP mixture string, e.g. "CO2;Methane".
            The order of components in this string defines the order in
            mole fraction arrays `x` and `y`.
        T : float
            Mixture temperature [K].
        P : list[float]
            Pressures [bar] at which equilibrium states are defined.
            Must have the same length as `x`, `y`, `rhoL`, `rhoV`,
            `M_liq`, and `M_vap`.
        x : list[list[float]]
            Liquid-phase mole fractions for each state point.
            Each entry is a vector of length Ncomp (here N=2).
        y : list[list[float]]
            Vapor-phase mole fractions for each state point.
            Same shape as `x`.
        rhoL : list[float]
            Liquid molar densities [mol/cm3] at each state point.
        rhoV : list[float]
            Vapor molar densities [mol/cm3] at each state point.
        kij : float, optional
            Binary interaction parameter for the Parachor mixing rule
            (default = 0.0).
        phi_ij : float or array, optional
            Mixing parameter for the WSD model.
            If scalar, the same value is used for all cross terms.
            If array, must be an NxN matrix.
        verbose : bool, optional
            If True, print debug information for each state point.

        Returns
        -------
        results : dict
            Dictionary with keys:
                - "P_bar"           : pressures [bar]
                - "gamma_parachor"  : IFT values [mN/m] from Parachor method
                - "gamma_WSD"       : IFT values [mN/m] from WSD method
                - "x"               : liquid mole fractions
                - "y"               : vapor mole fractions

        Notes
        -----
        - Input arrays must all have the same length (number of state points).
        - Compositions x and y must each sum to unity for every point.
        - Densities should already be converted to mol/cm3 before calling.
        """
        
        components      = mixture.split(";")
        ncomp           = len(components)
        npts            = len(P) 
              
        # --- Guard: outer lengths
        if not (len(x) == len(y) == len(rhoL) == len(rhoV) == npts):
            raise ValueError(
                f"Mismatched input lengths: "
                f"P={len(P)}, x={len(x)}, y={len(y)}, rhoL={len(rhoL)}, rhoV={len(rhoV)}")

        # --- Guard: inner lengths for compositions + sums
        for i, (xi, yi) in enumerate(zip(x, y)):
            if len(xi) != ncomp or len(yi) != ncomp:
                raise ValueError(
                    f"At point {i}: composition length mismatch. "
                    f"Expected {ncomp}, got len(x[{i}])={len(xi)}, len(y[{i}])={len(yi)}"
                )
            if abs(sum(xi) - 1.0) > 1e-6:
                raise ValueError(f"Liquid mole fractions at point {i} do not sum to 1 (sum={sum(xi)})")
            if abs(sum(yi) - 1.0) > 1e-6:
                raise ValueError(f"Vapor mole fractions at point {i} do not sum to 1 (sum={sum(yi)})")
            
        # --- per-component molar masses & parachor numbers at Teff (from pure gamma)
        MOLAR_MASSES        = []
        parachor_numbers    = []
        
        for comp in components:
            
            MM_g_per_mol = self.RP.REFPROPdll(comp, "TQ", "M", self.SI_BASE, 0, 0, T, 0, [1.0]).Output[0]
            MOLAR_MASSES.append(MM_g_per_mol)

            # Pure gamma (may switch to Teff=0.9Tc if T>Tc)
            gamma_i_mNpm, Teff = self.compute_gamma_pure(T, comp, key='Parachor', Tc=None)

            liq = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 0, [1.0])
            vap = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 1, [1.0])

            rhoL_kg_m3 = liq.Output[0]
            rhoV_kg_m3 = vap.Output[0]

            # Convert to mol/cm^3:
            # mol/m^3 = (kg/m^3) / (kg/mol) = (kg/m^3) / (MM_g/mol * 1e-3)
            # mol/cm^3 = previous / 1e6  => combined: / (MM_g_per_mol * 1e3)
            rhoL_mol_cm3_pure = rhoL_kg_m3 / (MM_g_per_mol * 1e3)
            rhoV_mol_cm3_pure = rhoV_kg_m3 / (MM_g_per_mol * 1e3)

            P_i = self.parachor_number(rhoL_mol_cm3_pure, rhoV_mol_cm3_pure, gamma_i_mNpm)
            parachor_numbers.append(P_i)

        parachor_numbers = np.asarray(parachor_numbers, dtype=float)
            

        # --- per-point IFTs
        P       = np.asarray(P, dtype=float)
        x_arr   = np.asarray(x, dtype=float)
        y_arr   = np.asarray(y, dtype=float)
        rhoL    = np.asarray(rhoL, dtype=float)
        rhoV    = np.asarray(rhoV, dtype=float)

        # Build a validity mask for each state point
        good    = np.isfinite(P) & np.all(np.isfinite(x_arr), axis=1) & np.all(np.isfinite(y_arr), axis=1) \
            & np.isfinite(rhoL) & np.isfinite(rhoV)
            
        if not np.any(good):
            # nothing to compute
            return {
                "P_bar": np.array([]),
                "gamma_parachor": np.array([]),
                "gamma_WSD": np.array([]),
                "x": np.empty((0, ncomp)),
                "y": np.empty((0, ncomp)),
                "rhoL": np.array([]),
                "rhoV": np.array([]),
            }

        P_f     = P[good]
        x_f     = x_arr[good, :]
        y_f     = y_arr[good, :]
        rhoL_f  = rhoL[good]
        rhoV_f  = rhoV[good]

        gamma_par = np.empty_like(P_f)
        gamma_wsd = np.empty_like(P_f)
        
        for i in range(len(P_f)):
            xi      = x_f[i, :].tolist()
            yi      = y_f[i, :].tolist()
            rhoLi   = float(rhoL_f[i])
            rhoVi   = float(rhoV_f[i])

            # Parachor
            gamma_par[i] = self.compute_gamma_Parachor(
                x=xi, y=yi,rho_l=rhoLi, rho_v=rhoVi,
                parachor_numbers=parachor_numbers,kij=kij)

            # WSD
            gamma_wsd[i] = self.compute_gamma_WSD(
                T=T, comp=components,x=xi, y=yi,
                rho_l=rhoLi, rho_v=rhoVi,phi=phi_ij)

            if verbose:
                print(f"[{i}] P={P_f[i]:.3f} bar | gamma_P={gamma_par[i]:.6f} mN/m | gamma_WSD={gamma_wsd[i]:.6f} mN/m")   
        
        # --- Sort by pressure (nice, ordered outputs)
        idx_sort = np.argsort(P_f)
        P_f       = P_f[idx_sort]
        x_f       = x_f[idx_sort]
        y_f       = y_f[idx_sort]
        rhoL_f    = rhoL_f[idx_sort]
        rhoV_f    = rhoV_f[idx_sort]
        gamma_par = gamma_par[idx_sort]
        gamma_wsd = gamma_wsd[idx_sort]

        # --- Enforce monotone-decreasing gamma(P): drop bumps beyond tolerances
        if enforce_monotone:
            keep_par, _ = self.decreasing_mask(P_f, gamma_par, rel_drop=0.20, abs_drop=0.2, near_zero=1e-3)
            keep_wsd, _ = self.decreasing_mask(P_f, gamma_wsd, rel_drop=0.20, abs_drop=0.2, near_zero=1e-3)
            keep = keep_par & keep_wsd

            P_f       = P_f[keep]
            x_f       = x_f[keep]
            y_f       = y_f[keep]
            rhoL_f    = rhoL_f[keep]
            rhoV_f    = rhoV_f[keep]
            gamma_par = gamma_par[keep]
            gamma_wsd = gamma_wsd[keep]

        return {
            "P_bar": P_f,
            "gamma_Parachor_mNpm": gamma_par,
            "gamma_WSD_mNpm": gamma_wsd,
            "x": x_f,
            "y": y_f,
            "rhoL": rhoL_f,
            "rhoV": rhoV_f,
        }
        
    def TP_DIAGRAM(self, mixture: str, z: list[float], T: float, verbose: bool = False):
        """
        Compute the TP phase envelope point (at fixed T) for a binary mixture using REFPROP.

        Parameters
        ----------
        mixture : str
            REFPROP mixture string, e.g. "CO2;Methane".
            The order of components in this string defines the order in mole fraction array `z`.
        z : list[float]
            Overall composition (same order as `mixture` components). Must sum to unity.
        T : float
            Temperature [K].
        verbose : bool, optional
            If True, print debug information.

        Returns
        -------
        out : dict
            {
            'bubble': {'x', 'y', 'P_bar'},
            'dew'   : {'x', 'y', 'P_bar'}
            }
            - Bubble point (Q=0): liquid overall; returns vapor-in-equilibrium 'y'.
            Here, x ≈ z, y = res_liq.y
            - Dew point (Q=1): vapor overall; returns liquid-in-equilibrium 'x'.
            Here, y ≈ z, x = res_vap.x

        Notes
        -----
        - Uses REFPROP's TQ flash at Q=0 and Q=1.
        - Pressure unit conversion to bar follows your existing convention (*10).
        """
        comps = mixture.split(';')
        if len(comps) != 2:
            raise ValueError("TP_DIAGRAM supports binary mixtures only, e.g. 'CO2;Methane'.")

        if abs(sum(z) - 1.0) > 1e-6:
            raise ValueError(f"Overall mole fractions do not sum to 1 (sum={sum(z)})")

        z = np.asarray(z, dtype=float)

        # --- Critical T check ---
        try:
            Tc = self.RP.REFPROPdll(mixture, "C", "TCRIT", self.SI_BASE, 0, 0, 0, 0, z.tolist()).Output[0]
        except Exception as e:
            raise RuntimeError(f"Failed to get Tc for {mixture} from REFPROP.") from e

        if T >= Tc:
            raise ValueError(f"TP_DIAGRAM requires T < Tc (here T={T} K >= Tc={Tc} K).")

        self.RP.FLAGSdll('GERG', 1)
        
        # --- Saturation states at given T ---
        # Q = 0 -> saturated liquid (bubble point)
        res_liq = self.RP.REFPROPdll(mixture, "TQ", "D;P;STN", self.SI_BASE, 0, 0, T, 0.0, z)
        # Q = 1 -> saturated vapor (dew point)
        res_vap = self.RP.REFPROPdll(mixture, "TQ", "D;P;STN", self.SI_BASE, 0, 0, T, 1.0, z)

        # Pressures (keep your conversion factor)
        P_bubble_bar = res_liq.Output[1] * 10.0  # bubble: Q=0
        P_dew_bar    = res_vap.Output[1] * 10.0  # dew:    Q=1

        # Equilibrium compositions
        # At bubble: overall is liquid -> x ≈ z, y from res_liq.y
        x_bubble = z.tolist()
        y_bubble = list(res_liq.y)  # vapor in eq. with sat. liquid

        # At dew: overall is vapor -> y ≈ z, x from res_vap.x
        x_dew = list(res_vap.x)     # liquid in eq. with sat. vapor
        y_dew = z.tolist()

        if verbose:
            print(f"T = {T} K")
            print(f"Critical Temperature (Tc): {Tc:.4f} K")
            print(f"Bubble: P = {P_bubble_bar:.6g} bar, x = {x_bubble}, y = {y_bubble}")
            print(f"Dew   : P = {P_dew_bar:.6g} bar, x = {x_dew}, y = {y_dew}")

        return {
            'Tc': Tc,
            'bubble': {'x': x_bubble, 'y': y_bubble, 'P_bar': P_bubble_bar},
            'dew'   : {'x': x_dew,    'y': y_dew,    'P_bar': P_dew_bar}
        }