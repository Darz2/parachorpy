#!/usr/bin/env python

# Created by Darshan on 2025-07-15
# LAST UPDATED: 2025-03-09 by Darshan (Fixed some bugs and genralize the TP diagram to n components)

import json,os,re,math,numpy as np,pandas as pd, collections.abc
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from pathlib import Path
from scipy.interpolate import PchipInterpolator

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

    def get_tmin_ttrp_tc(self, fluid_name: str):
        """
        Return Tmin, Ttrp, and Tc [K] for a pure fluid from REFPROP.
        """
        res = self.RP.REFPROPdll(
            fluid_name, "", "TMIN;TTRP;TC", self.SI_BASE, 0, 0, 0, 0, [1.0]
        )
        return res.Output[0], res.Output[1], res.Output[2]

    def compute_gamma_pure(self, T, fluid_name, key, Tc=None):
        """
        Compute Interfacial tension (IFT) for a pure fluid at temperature T.

        For Parachor:
            T_used = min(max(T, max(TMIN, 1.1*TTRP)), 0.9*TC)

        For WSD:
            use T directly if T <= TC, otherwise gamma = 0

        Returns
        -------
        gamma : float
            Interfacial tension [mN/m]
        T_used : float
            Temperature actually used in the correlation [K]
        """
        _, s, n = self.get_fluid_params(fluid_name)

        if len(s) != len(n):
            raise ValueError("Length of s and n must be equal")
        
        Tmin, Ttrp, Tc_final = self.get_tmin_ttrp_tc(fluid_name)

        if key == 'Parachor':
            
            if T <= Tc_final and T >= max(Tmin, Ttrp):
                gamma_T = sum(si * (1 - T / Tc_final)**ni for si, ni in zip(s, n))
                T_used = T

            if T > Tc_final:
                print(f"Given temperature is greater than Tc ({Tc_final}) for {fluid_name}")
                print(f"Computing interfacial tension at T = 0.9*{Tc_final} for Parachor method")
                T_used  = 0.9 * Tc_final
                gamma_T = sum(si * (1 - T_used / Tc_final)**ni for si, ni in zip(s, n))
            
            if T < max(Tmin, Ttrp): # This only happens for computing the parachor number in case of mixtures
                print(f"Given temperature is less than or equal to max(TMIN, TTRP) for {fluid_name}")
                print(f"Computing interfacial tension at T = 1.1*max(TMIN, TTRP) for Parachor method for mixtures")
                T_used = 1.1 * max(Tmin, Ttrp)
                gamma_T = sum(si * (1 - T_used / Tc_final)**ni for si, ni in zip(s, n))
            
        
        elif key == "WSD":
            if T <= Tc_final:
                gamma_T = sum(si * (1 - T / Tc_final)**ni for si, ni in zip(s, n))
                T_used = T
            
            if T > Tc_final:
                gamma_T = 0
                T_used = T
        
        else:
            raise ValueError("The key should be either Parachor or WSD")

        return gamma_T * 1e3, T_used  # Convert to mN/m

    def parachor_number(self,rho_l, rho_v, gamma):
        """
        Calculate the parachor number.

        Parameters:
        - rho_l : float — Liquid density [kg/m3]
        - rho_v : float — Vapor density [kg/m3]
        - gamma : float — Interfacial tension [mN/m]

        Returns:
        - P : float — parachor number [kg/m3]n ·mN/m

        Constant:
        - n : float — exponent in the parachor equation (default: 3.87); REFPROP v10 (Lemmon et al, 2018)
        """
        n = 3.87
        if rho_l <= 0 or rho_v <= 0 or gamma <= 0:
            raise ValueError("All inputs must be positive")

        delta_rho = rho_l - rho_v               # [mol/cm3]
        gamma_SI  = gamma                       # in  mN/m

        parachor  = gamma_SI**(1/n) / delta_rho
        print(f"Parachor number calculated: {parachor:.6f} [kg/m3]n ·mN/m")
        
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

        delta_parachor     = rho_l * P_l - rho_v * P_v
    
        if delta_parachor <= 0:
            return np.nan
        
        gamma_mix_Parachor = delta_parachor ** n
        
        return gamma_mix_Parachor
    
    # Created by Darshan on 2025-10-01
    def compute_gamma_WSD(self, T, comp, x, y, rho_l, rho_v, phi=1.0, correction=True):
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
            return 0.0, 1.0  # nothing contributes above all critical temps
    
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
        
        # ---- Step 5: Supercritical correction (Roar correction) ----
        
        mixcorr = 1.0
        if correction and not all(active): 
            x_inactive_sum = float(np.sum(x[[i for i in range(N) if not active[i]]]))
            mixcorr = max(0.0, 1.0 - x_inactive_sum)   # guard against tiny negatives
            gamma_mix_WSD *= mixcorr
        
        return gamma_mix_WSD, mixcorr
        
    def gamma_PT(self, mixture: str, z: list[float], T: float, pressures_bar: float | list[float],
                        kij: float = 0.0, phi_ij=1.0, correction=True):
        """
        Compute the interfacial tension (IFT) of a mixture along its
        phase envelope (PT diagram) at fixed temperature.
        
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
        correction : bool
            Whether to apply supercritical correction only to the WSD model (default: True). If True, WSD results will be multiplied by (1 - sum(x_inactive)),

        Returns
        -------
        pressures_bar           : list[float]
        results_gamma_Parachor  : list[float]
        results_gamma_WSD       : list[float]
        results_mixcorr_WSD     : list[float]
        results_xL              : list[np.ndarray]
        results_yV              : list[np.ndarray]
        results_rhoL_kg_m3      : list[float]
        results_rhoV_kg_m3      : list[float]
        results_MM_L            : list[float]
        results_MM_V            : list[float]
        """

        # Allow single float or iterable
        if isinstance(pressures_bar, collections.abc.Iterable) and not isinstance(pressures_bar, (str, bytes)):
            pressures_bar = list(pressures_bar)
        else:
            pressures_bar = [pressures_bar]
        
        components      = [c.strip() for c in mixture.split(";") if c.strip()]
        ncomp           = len(components)
        z               = np.asarray(z, dtype=float)
        
        if len(z) != ncomp:
            raise ValueError(
                f"Composition length mismatch: mixture has {ncomp} components, but z has length {len(z)}.")

        if abs(np.sum(z) - 1.0) > 1e-6:
            raise ValueError(f"Overall mole fractions do not sum to 1 (sum={np.sum(z)})")
        
        pressures_bar   = list(pressures_bar)

        # --- per-component molar masses & parachor numbers at Teff (from pure gamma)
        MOLAR_MASSES        = []
        parachor_numbers    = []

        for comp in components:
            
            MM_res = self.RP.REFPROPdll(comp, "TQ", "M", self.SI_BASE, 0, 0, T, 0, [1.0])
            if MM_res.ierr > 0:
                raise RuntimeError(f"Failed molar mass query for {comp}: {MM_res.herr.strip()}")
            MM = MM_res.Output[0]
            MOLAR_MASSES.append(MM)

            # Pure gamma (may switch to Teff=0.9Tc if T>Tc)
            gamma_val_mNpm, Teff = self.compute_gamma_pure(T, comp, key='Parachor', Tc=None)

            liq = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 0, [1.0])
            vap = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 1, [1.0])
            
            if liq.ierr > 0:
                raise RuntimeError(f"Failed liquid saturation query for {comp}: {liq.herr.strip()}")
            if vap.ierr > 0:
                raise RuntimeError(f"Failed vapor saturation query for {comp}: {vap.herr.strip()}")

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
        results_mixcorr_WSD     = []
        
        results_MM_L            = []
        results_MM_V            = []
        results_rhoL_kg_m3      = []
        results_rhoV_kg_m3      = []   
        results_xL              = []
        results_yV              = []     
        
        for Pbar in pressures_bar:
            
            mix = self.RP.REFPROPdll(mixture, "TP", "D;Dliq;Dvap", self.SI_BASE, 0, 1, T, Pbar / 10.0, z.tolist())
            
            if mix.ierr > 0:
                raise RuntimeError(f"TP flash failed at T={T}, P={Pbar} bar: {mix.herr.strip()}")
            
            xL  = np.array(mix.x[:ncomp], dtype=float)
            yV  = np.array(mix.y[:ncomp], dtype=float)

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

            gamma_mix_Parachor      = self.compute_gamma_Parachor(
                                        x=xL.tolist(),y=yV.tolist(),
                                        rho_l=rhoL_mol_cm3,rho_v=rhoV_mol_cm3,
                                        parachor_numbers=parachor_numbers,kij=kij)
            
            gamma_mix_WSD, mixcorr  = self.compute_gamma_WSD(
                                        T=T, comp = components,x=xL.tolist(),
                                        y=yV.tolist(),rho_l=rhoL_mol_cm3,
                                        rho_v=rhoV_mol_cm3,phi=phi_ij, correction=correction)
            
            results_gamma_Parachor.append(gamma_mix_Parachor)
            results_gamma_WSD.append(gamma_mix_WSD)
            results_mixcorr_WSD.append(mixcorr)
            
        return pressures_bar, results_gamma_Parachor, results_gamma_WSD,  results_mixcorr_WSD, \
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
            
            print(rhoL_mol_cm3_pure, rhoV_mol_cm3_pure, gamma_i_mNpm)

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

        gamma_par   = np.empty_like(P_f)
        gamma_wsd   = np.empty_like(P_f)
        mixcorr_wsd = np.empty_like(P_f)
        
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
            gamma_wsd[i], mixcorr_wsd[i] = self.compute_gamma_WSD(
                T=T, comp=components,x=xi, y=yi,
                rho_l=rhoLi, rho_v=rhoVi,phi=phi_ij)

            if verbose:
                print(f"[{i}] P={P_f[i]:.3f} bar | gamma_P={gamma_par[i]:.6f} mN/m | gamma_WSD={gamma_wsd[i]:.6f} mN/m | mixcorr={mixcorr_wsd[i]:.6f}")   
        
        # --- Sort by pressure (nice, ordered outputs)
        idx_sort    = np.argsort(P_f)
        P_f         = P_f[idx_sort]
        x_f         = x_f[idx_sort]
        y_f         = y_f[idx_sort]
        rhoL_f      = rhoL_f[idx_sort]
        rhoV_f      = rhoV_f[idx_sort]
        gamma_par   = gamma_par[idx_sort]
        gamma_wsd   = gamma_wsd[idx_sort]
        mixcorr_wsd = mixcorr_wsd[idx_sort]

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
            mixcorr_wsd = mixcorr_wsd[keep]

        return {
            "P_bar": P_f,
            "gamma_Parachor_mNpm": gamma_par,
            "gamma_WSD_mNpm": gamma_wsd,
            "mixcorr_WSD": mixcorr_wsd,
            "x": x_f,
            "y": y_f,
            "rhoL": rhoL_f,
            "rhoV": rhoV_f,
        }
        
    def TP_DIAGRAM(self, mixture: str, z: list[float], T: float, verbose: bool = False):
        """
            Compute the TP phase-envelope pressures (at fixed T) for a mixture using REFPROP.

            Parameters
            ----------
            mixture : str
                REFPROP mixture string, e.g. "CO2;Methane" or "Methane;Ethane;Propane".
                The order of components defines the order in mole fraction array `z`.
            z : list[float]
                Overall mole-fraction composition. Must sum to unity.
            T : float
                Temperature [K]. Must be less than the mixture critical temperature Tc.
            verbose : bool, optional
                If True, print debug information.

        Returns
        -------
        out : dict
            Dictionary containing critical point data and phase envelope points:
            {
                'Tc': float,                    # Critical temperature [K]
                'Pc': float,                    # Critical pressure [bar]
                'bubble': {
                    'x': list[float],           # Liquid mole fraction
                    'y': list[float],           # Vapor mole fraction
                    'P_bar': float              # Bubble point pressure [bar] or np.nan
                },
                'dew': {
                    'x': list[float],           # Liquid mole fraction
                    'y': list[float],           # Vapor mole fraction
                    'P_bar': float              # Dew point pressure [bar] or np.nan
                }
            }

        Notes
        -----
        - Uses REFPROP's TQ flash at Q=0 (bubble: saturated liquid) and Q=1 (dew: saturated vapor).
        - Uses SATSPLN + TC;PC to obtain the mixture critical point.
        - Negative ierr values from REFPROP are warnings, not fatal errors.
        - Near the critical point, TQ flashes may fail to converge; in that case NaN values are returned.
        - Pressure conversion is taken here as MPa -> bar using factor 10

        Raises
        ------
        ValueError
            If T >= mixture critical temperature or if overall composition does not sum to unity.
        RuntimeError
            If REFPROP fails to return the mixture critical point.
        """
        comps = [c.strip() for c in mixture.split(';') if c.strip()]
        ncomp = len(comps)

        if len(z) != ncomp:
            raise ValueError(
                f"Composition length mismatch: mixture has {ncomp} components, but z has length {len(z)}."
            )

        if abs(sum(z) - 1.0) > 1e-6:
            raise ValueError(f"Overall mole fractions do not sum to 1 (sum={sum(z)})")

        z = np.asarray(z, dtype=float)

        # Enable GERG
        self.RP.FLAGSdll('GERG', 1)

        # --- Mixture critical point ---
        try:
            sat = self.RP.REFPROPdll(mixture, "SATSPLN", " ", self.SI_BASE, 0, 0, 0, 0, z.tolist())

            if sat.ierr > 0:
                raise RuntimeError(
                    f"SATSPLN failed: ierr={sat.ierr}, herr='{sat.herr.strip()}'")
            elif sat.ierr < 0 and verbose:
                print(f"SATSPLN warning: ierr={sat.ierr}, herr='{sat.herr.strip()}'")

            crit = self.RP.REFPROPdll(mixture, "", "TC;PC", self.SI_BASE, 0, 0, 0, 0, z.tolist())

            if crit.ierr > 0:
                raise RuntimeError(
                    f"Critical-point query failed: ierr={crit.ierr}, herr='{crit.herr.strip()}'")
            elif crit.ierr < 0 and verbose:
                print(f"Critical-point warning: ierr={crit.ierr}, herr='{crit.herr.strip()}'")

            Tc = crit.Output[0]         # K
            Pc = crit.Output[1] * 10.0  # MPa -> bar

        except Exception as e:
            raise RuntimeError(f"Failed to get mixture critical point for {mixture}.") from e

        if T >= Tc:
            raise ValueError(f"TP_DIAGRAM requires T < Tc (here T={T} K >= Tc={Tc} K).")

        # --- Default outputs as NaN ---
        bubble = {
            'x': [np.nan] * ncomp,
            'y': [np.nan] * ncomp,
            'P_bar': np.nan
        }

        dew = {
            'x': [np.nan] * ncomp,
            'y': [np.nan] * ncomp,
            'P_bar': np.nan
        }

        # --- Bubble point: Q = 0 ---
        try:
            res_liq = self.RP.REFPROPdll(
                mixture, "TQ", "D;P;STN", self.SI_BASE, 0, 0, T, 0.0, z.tolist())

            if res_liq.ierr > 0:
                raise RuntimeError(
                    f"Bubble-point calculation failed: ierr={res_liq.ierr}, herr='{res_liq.herr.strip()}'")
            elif res_liq.ierr < 0 and verbose:
                print(f"Bubble-point warning: ierr={res_liq.ierr}, herr='{res_liq.herr.strip()}'")

            bubble = {
                'x': z.tolist(),
                'y': list(res_liq.y[:ncomp]),
                'P_bar': res_liq.Output[1] * 10.0
            }

        except Exception as e:
            if verbose:
                print(f"Bubble-point skipped at T={T:.6f} K: {e}")

        # --- Dew point: Q = 1 ---
        try:
            res_vap = self.RP.REFPROPdll(
                mixture, "TQ", "D;P;STN", self.SI_BASE, 0, 0, T, 1.0, z.tolist()
            )

            if res_vap.ierr > 0:
                raise RuntimeError(
                    f"Dew-point calculation failed: ierr={res_vap.ierr}, herr='{res_vap.herr.strip()}'"
                )
            elif res_vap.ierr < 0 and verbose:
                print(f"Dew-point warning: ierr={res_vap.ierr}, herr='{res_vap.herr.strip()}'")

            dew = {
                'x': list(res_vap.x[:ncomp]),
                'y': z.tolist(),
                'P_bar': res_vap.Output[1] * 10.0
            }

        except Exception as e:
            if verbose:
                print(f"Dew-point skipped at T={T:.6f} K: {e}")

        if verbose:
            print(f"T = {T:.6f} K")
            print(f"Critical Temperature (Tc): {Tc:.6f} K")
            print(f"Critical Pressure    (Pc): {Pc:.6f} bar")
            print(f"Bubble: P = {bubble['P_bar']}, x = {bubble['x']}, y = {bubble['y']}")
            print(f"Dew   : P = {dew['P_bar']}, x = {dew['x']}, y = {dew['y']}")

        return {
            'Tc': Tc,
            'Pc': Pc,
            'bubble': bubble,
            'dew': dew
        }

    def PT_envelope_curves(self, csv_path: str, n_smooth: int = 400):
        """
        Read a PT-envelope CSV file, clean the data, append the critical point,
        and return smoothed bubble/dew curves using PCHIP interpolation.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        n_smooth : int, optional
            Number of points for the smoothed curves.

        Returns
        -------
        out : dict
            {
                "mixture": str,
                "z_label": str,
                "Tc": float,
                "Pc": float,
                "T_bub_raw": np.ndarray,
                "P_bub_raw": np.ndarray,
                "T_dew_raw": np.ndarray,
                "P_dew_raw": np.ndarray,
                "T_bub_smooth": np.ndarray,
                "P_bub_smooth": np.ndarray,
                "T_dew_smooth": np.ndarray,
                "P_dew_smooth": np.ndarray,
            }
        """
        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("CSV file is empty.")

        required_cols = {"mixture", "point_type", "T_K", "Tc_K", "Pc_bar", "P_bub_bar", "P_dew_bar"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV file is missing required columns: {sorted(missing)}")

        # --- metadata ---
        mixture = str(df["mixture"].iloc[0])
        Tc = float(df["Tc_K"].iloc[0])
        Pc = float(df["Pc_bar"].iloc[0])

        z_cols = [c for c in df.columns if c.startswith("z_")]
        if z_cols:
            comp_names = [c[2:] for c in z_cols]
            z_vals = [float(df[c].iloc[0]) for c in z_cols]
            z_label = ", ".join(f"{name}={val:.4f}" for name, val in zip(comp_names, z_vals))
        else:
            z_label = "composition unavailable"

        # --- clean bubble branch ---
        df_bub_clean = (
            df[df["point_type"] == "bubble"]
            .copy()
            .loc[lambda d: d["P_bub_bar"] > 0]
            .drop_duplicates(subset=["T_K"])
            .sort_values("T_K")
            .reset_index(drop=True)
        )

        # --- clean dew branch ---
        df_dew_clean = (
            df[df["point_type"] == "dew"]
            .copy()
            .loc[lambda d: d["P_dew_bar"] > 0]
            .drop_duplicates(subset=["T_K"])
            .sort_values("T_K")
            .reset_index(drop=True)
        )

        if df_bub_clean.empty:
            raise ValueError("No valid bubble-point data found.")
        if df_dew_clean.empty:
            raise ValueError("No valid dew-point data found.")

        # --- append critical point ---
        T_bub_raw = np.append(df_bub_clean["T_K"].to_numpy(dtype=float), Tc)
        P_bub_raw = np.append(df_bub_clean["P_bub_bar"].to_numpy(dtype=float), Pc)

        T_dew_raw = np.append(df_dew_clean["T_K"].to_numpy(dtype=float), Tc)
        P_dew_raw = np.append(df_dew_clean["P_dew_bar"].to_numpy(dtype=float), Pc)

        # --- sort ---
        bubble_pts = np.column_stack([T_bub_raw, P_bub_raw])
        bubble_pts = bubble_pts[np.argsort(bubble_pts[:, 0])]

        dew_pts = np.column_stack([T_dew_raw, P_dew_raw])
        dew_pts = dew_pts[np.argsort(dew_pts[:, 0])]

        T_bub_raw, P_bub_raw = bubble_pts[:, 0], bubble_pts[:, 1]
        T_dew_raw, P_dew_raw = dew_pts[:, 0], dew_pts[:, 1]

        # --- shape-preserving interpolation ---
        T_bub_smooth = np.linspace(T_bub_raw.min(), T_bub_raw.max(), n_smooth)
        T_dew_smooth = np.linspace(T_dew_raw.min(), T_dew_raw.max(), n_smooth)

        P_bub_smooth = PchipInterpolator(T_bub_raw, P_bub_raw)(T_bub_smooth)
        P_dew_smooth = PchipInterpolator(T_dew_raw, P_dew_raw)(T_dew_smooth)

        return {
            "mixture": mixture,
            "z_label": z_label,
            "Tc": Tc,
            "Pc": Pc,
            "T_bub_raw": T_bub_raw,
            "P_bub_raw": P_bub_raw,
            "T_dew_raw": T_dew_raw,
            "P_dew_raw": P_dew_raw,
            "T_bub_smooth": T_bub_smooth,
            "P_bub_smooth": P_bub_smooth,
            "T_dew_smooth": T_dew_smooth,
            "P_dew_smooth": P_dew_smooth,
        }
        
    def compute_IFT_envelope(self, mixture: str, z: list[float], T_start: float = 200,
                          N_points: int = 20, kij: float = 0.0, phi_ij=1.0,
                          tol_frac: float = 5e-6, mode: str = "envelope",
                          verbose: bool = False):
        
        """
        Sweep temperature from T_start up to the mixture critical point and compute
        interfacial tension (Parachor + WSD) at each temperature.

        Parameters
        ----------
        mixture   : str         REFPROP mixture string, e.g. "CO2;Methane"
        z         : list[float] Overall mole fractions (must sum to 1)
        T_start   : float       Starting temperature [K] (default: 200)
        N_points  : int         Number of pressure points between P_bub and P_dew.
                                Only used when mode="envelope" (default: 20)
        kij       : float       Binary interaction parameter for Parachor (default: 0.0)
        phi_ij    : float       WSD cross-parameter (default: 1.0)
        tol_frac  : float       Fractional inset from saturation pressures (default: 5e-6)
        mode      : str         Pressure sampling strategy:
                                "saturation" — evaluate only at bubble and dew points (2 points per T)
                                "envelope"   — sweep N_points pressures inside the two-phase region

        Returns
        -------
        df     : pd.DataFrame   Full results (all temperatures and pressures)
        df_bub : pd.DataFrame   Subset closest to bubble point (first pressure at each T)
        df_dew : pd.DataFrame   Subset closest to dew point (last pressure at each T)
        """

        if mode not in ("saturation", "envelope"):
            raise ValueError(f"mode must be 'saturation' or 'envelope', got '{mode}'.")

        comps = [c.strip() for c in mixture.split(';') if c.strip()]
        ncomp = len(comps)

        if len(z) != ncomp:
            raise ValueError(
                f"Composition length mismatch: mixture has {ncomp} components "
                f"but z has {len(z)} entries."
            )
        if abs(sum(z) - 1.0) > 1e-6:
            raise ValueError(f"Overall mole fractions do not sum to 1 (sum={sum(z):.6f})")

        # Critical point
        crit_result = self.TP_DIAGRAM(mixture, z, float(T_start), verbose=verbose)
        Tc = crit_result['Tc']
        Pc = crit_result['Pc']
        print(f"Mixture critical point:  Tc = {Tc:.4f} K  |  Pc = {Pc:.4f} bar")
        print(f"Mode: '{mode}'" + (f" | N_points = {N_points}" if mode == "envelope" else " | 2 points per T (bubble + dew)"))

        T_list = np.arange(T_start, int(np.floor(Tc)), 1).tolist()
        rows   = []

        # Temperature sweep 
        for T in T_list:

            result = self.TP_DIAGRAM(mixture, z, T, verbose=verbose)
            P_bub  = result['bubble']['P_bar']
            P_dew  = result['dew']['P_bar']
            Tc_loc = result['Tc']
            Pc_loc = result['Pc']

            if np.isnan(P_bub) or np.isnan(P_dew):
                print(f"  [skip] T = {T:.1f} K — saturation branch unavailable.")
                continue

            # Pressure grid: choose based on mode 
            if mode == "saturation":
                # Only the two saturation boundaries
                PRESSURES = np.array([
                    P_bub * (1.0 - tol_frac),
                    P_dew * (1.0 + tol_frac),
                ])
            else:  # mode == "envelope"
                # Full sweep across the two-phase interior
                PRESSURES = np.linspace(
                    P_bub * (1.0 - tol_frac),
                    P_dew * (1.0 + tol_frac),
                    N_points
                )

            try:
                Pbar, gamma_Parachor, gamma_WSD, mixcorr_WSD, \
                    x, y, rhol, rhov, mml, mmv = \
                    self.gamma_PT(mixture, z, T, PRESSURES, kij, phi_ij, correction=True)
            except Exception as exc:
                print(f"  [skip] T = {T:.1f} K — gamma_PT failed: {exc}")
                continue

            for P, s1, s2, s3, xl, yv, ml, mv, rhol_i, rhog_i in zip(
                Pbar, gamma_Parachor, gamma_WSD, mixcorr_WSD,
                x, y, mml, mmv, rhol, rhov
            ):
                # point_type is meaningful for both modes
                point_type = "bubble" if abs(P - P_bub) <= abs(P - P_dew) else "dew"

                row = {
                    "mixture"             : mixture,
                    "T_K"                 : round(float(T),       4),
                    "Tc_K"                : round(float(Tc_loc),  6),
                    "Pc_bar"              : round(float(Pc_loc),  6),
                    "kij"                 : kij,
                    "phi_ij"              : phi_ij,
                    "mode"                : mode,
                    "point_type"          : point_type,
                    "P_bar"               : round(float(P),       6),
                    "P_bub_bar"           : round(float(P_bub),   6),
                    "P_dew_bar"           : round(float(P_dew),   6),
                    "gamma_Parachor_mN_m" : round(float(s1),      8),
                    "gamma_WSD_mN_m"      : round(float(s2),      8),
                    "mixcorr_WSD"         : round(float(s3),      8),
                    "M_liq_g_per_mol"     : round(float(ml),      8),
                    "M_vap_g_per_mol"     : round(float(mv),      8),
                    "rho_liq_kg_per_m3"   : round(float(rhol_i),  6),
                    "rho_vap_kg_per_m3"   : round(float(rhog_i),  6),
                }

                for comp, zi in zip(comps, z):
                    row[f"z_{comp}"] = round(float(zi), 6)
                for comp, xi in zip(comps, xl):
                    row[f"x_{comp}"] = round(float(xi), 6)
                for comp, yi in zip(comps, yv):
                    row[f"y_{comp}"] = round(float(yi), 6)

                rows.append(row)

            if verbose:
                print(f"\n--- T = {T:.1f} K | Tc = {Tc_loc:.4f} K | Pc = {Pc_loc:.4f} bar ---")
                print(f"  P_bub = {P_bub:.6f} bar  |  P_dew = {P_dew:.6f} bar")
                print(f"  gamma_Parachor : {[round(float(v), 6) for v in gamma_Parachor]}")
                print(f"  gamma_WSD      : {[round(float(v), 6) for v in gamma_WSD]}")
                print(f"  mixcorr_WSD    : {[round(float(v), 6) for v in mixcorr_WSD]}")

        df = pd.DataFrame(rows)

        if df.empty:
            print("Warning: no valid rows collected.")
            empty = pd.DataFrame()
            return empty, empty, empty

        df_bub = df[df["point_type"] == "bubble"].reset_index(drop=True)
        df_dew = df[df["point_type"] == "dew"].reset_index(drop=True)

        print(f"\nDone — {len(df)} rows total  |  {len(df_bub)} bubble  |  {len(df_dew)} dew.")

        return df, df_bub, df_dew
    
    def plot_IFT_colormap(self, csv_path: str, model_key: str = "Parachor",
                        gamma_min_threshold: float = 1e-4, grid_size: int = 150,
                        save_path: str = None):
        """
        Create a single colormap plot of interfacial tension in PT space from a CSV file.
        Call once per model_key to get separate figures.

        Parameters
        ----------
        csv_path              : str    Path to the IFT results CSV file
        model_key             : str    "Parachor", "WSD", or "mixcorr" (default: "Parachor")
        gamma_min_threshold   : float  Filter out IFT values below this threshold (default: 1e-4)
        grid_size             : int    Resolution of the interpolation grid (default: 150)
        save_path             : str    Base filename (without extension) to save PNG + PDF.
                                    If None, figure is shown but not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        import matplotlib.pyplot       as plt 
        import thermoift.PLOT_SETTINGS as ps
        from scipy.interpolate         import griddata
        from matplotlib.path           import Path as MplPath
        from scipy.ndimage             import binary_erosion

        # --- Validate model_key ---
        valid_keys = ("Parachor", "WSD", "mixcorr")
        if model_key not in valid_keys:
            raise ValueError(f"model_key must be one of {valid_keys}, got '{model_key}'.")

        # --- Load CSV ---
        df = pd.read_csv(csv_path)

        required = {"T_K", "P_bar", "Tc_K", "Pc_bar",
                    "gamma_Parachor_mN_m", "gamma_WSD_mN_m",
                    "mixcorr_WSD", "P_bub_bar", "P_dew_bar"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        # --- Map model_key to column, colormap, colorbar label, format ---
        model_config = {
            "Parachor" : ("gamma_Parachor_mN_m", plt.cm.Blues,  r'$\gamma \; / \; [\mathrm{mN \, m^{-1}}]$', '%.0f'),
            "WSD"      : ("gamma_WSD_mN_m",      plt.cm.Blues,   r'$\gamma \; / \; [\mathrm{mN \, m^{-1}}]$', '%.0f'),
            "mixcorr"  : ("mixcorr_WSD",         plt.cm.Greys, r'$f_{\mathrm{corr}}$ / [-]',                 '%.2f'),
        }
        gamma_col, cmap, cbar_label, fmt = model_config[model_key]

        # --- Extract phase envelope directly from CSV (already computed) ---
        Tc     = float(df["Tc_K"].iloc[0])
        Pc     = float(df["Pc_bar"].iloc[0])
        env_df = (
            df[["T_K", "P_bub_bar", "P_dew_bar"]]
            .drop_duplicates(subset=["T_K"])
            .sort_values("T_K")
            .reset_index(drop=True)
        )
        T_bubble = env_df["T_K"].tolist()       + [Tc]
        P_bubble = env_df["P_bub_bar"].tolist() + [Pc]
        T_dew    = env_df["T_K"].tolist()       + [Tc]
        P_dew    = env_df["P_dew_bar"].tolist() + [Pc]

        envelope_path = MplPath(
            np.column_stack([T_bubble + T_dew[::-1],
                            P_bubble + P_dew[::-1]])
        )

        # --- Metadata ---
        mixture       = str(df["mixture"].iloc[0])
        comps         = [c.strip() for c in mixture.split(";")]
        z_cols        = [f"z_{c}" for c in comps]
        z_vals        = [float(df[zc].iloc[0]) for zc in z_cols if zc in df.columns]
        title_mixture = ps.format_mixture_latex(mixture)
        title_z       = ps.format_z_latex(mixture, z_vals)

        # --- Filter valid data ---
        threshold = gamma_min_threshold if model_key != "mixcorr" else -np.inf
        df_v      = df[df[gamma_col] >= threshold].copy()
        T_g       = df_v["T_K"].values
        P_g       = df_v["P_bar"].values
        g_data    = df_v[gamma_col].values

        if len(g_data) < 4:
            raise RuntimeError(
                f"Insufficient valid points ({len(g_data)}) for '{gamma_col}'. "
                f"Check gamma_min_threshold or CSV content."
            )

        # --- Grid + cubic interpolation ---
        T_grid         = np.linspace(T_g.min(), T_g.max(), grid_size)
        P_grid         = np.linspace(P_g.min(), P_g.max(), grid_size)
        T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

        g_mesh = griddata(
            (T_g, P_g), g_data,
            (T_mesh, P_mesh),
            method='cubic', fill_value=np.nan
        )

        # --- Mask outside the phase envelope ---
        inside   = envelope_path.contains_points(
            np.column_stack([T_mesh.ravel(), P_mesh.ravel()])
        ).reshape(T_mesh.shape)

        g_masked = np.ma.masked_where(~inside, g_mesh)
        g_eroded = np.ma.masked_where(~binary_erosion(inside, iterations=2), g_mesh)

        gmin = float(np.nanmin(g_masked))
        gmax = float(np.nanmax(g_masked))

        # --- Isoline levels ---
        levels_fill  = np.linspace(gmin, gmax, 35)
        if model_key == "mixcorr":
            levels_lines = [v for v in np.arange(0, 1.05, 0.1)   if gmin <= v <= gmax]
        else:
            levels_lines = [v for v in np.arange(0, gmax + 1, 2) if gmin <= v <= gmax]

        # --- Figure ---
        fig, ax = ps.plot_init()

        # filled contour
        contour = ax.contourf(
            T_mesh, P_mesh, g_masked,
            levels=levels_fill, cmap=cmap, extend='both'
        )

        # contour lines (erode mask slightly to avoid edge artifacts)
        if levels_lines:
            contour_lines = ax.contour(
                T_mesh, P_mesh, g_eroded,
                levels=levels_lines, colors='black',
                linewidths=ps.linewidth / 2, alpha=0.6
            )
            ax.clabel(
                contour_lines, levels=levels_lines,
                inline=True, fontsize=ps.label_fontsize / 2,
                fmt=fmt, inline_spacing=15, rightside_up=True
            )

        # colorbar
        cbar = fig.colorbar(contour, ax=ax, pad=0.02, format=fmt)
        cbar.set_ticks(levels_lines)
        cbar.set_label(cbar_label, fontsize=ps.label_fontsize)
        ps.style_colorbar(cbar)

        # phase envelope
        ax.plot(T_bubble, P_bubble, color="b", linewidth=ps.linewidth, linestyle="-",
                label="Bubble curve", zorder=10)
        ax.plot(T_dew,    P_dew,    color="r", linewidth=ps.linewidth, linestyle="-",
                label="Dew curve",    zorder=10)
        ax.plot([T_dew[-2], Tc], [P_dew[-2], Pc],
                linestyle="-", linewidth=ps.linewidth, color="red", zorder=10)

        # critical point
        label_cp = rf"($T_c={Tc:.1f}$ K, $P_c={Pc:.2f}$ bar)"
        ax.plot(Tc, Pc, "o", markersize=ps.markersize, zorder=15,
                markerfacecolor="grey", markeredgewidth=ps.markeredgewidth,
                markeredgecolor="black",
                label=rf"Critical point") # + " " + label_cp)

        # labels and formatting
        ax.set_xlabel(r'$T \; / \; [\mathrm{K}]$',  fontsize=ps.label_fontsize)
        ax.set_ylabel(r'$P \; / \; [\mathrm{bar}]$', fontsize=ps.label_fontsize)
        ax.set_title(f"{title_mixture} — {model_key}\n{title_z}",
                    fontsize=ps.label_fontsize / 2)
        ax.set_xlim(left=T_g.min() - 2)
        ax.minorticks_on()
        ps.style_legend(ax, fontsize=ps.legend_fontsize, loc='best', framealpha=0)

        plt.tight_layout()

        # --- Save ---
        if save_path is not None:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax