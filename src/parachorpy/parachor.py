#!/usr/bin/env python

# Created by Darshan on 2025-07-15

import json,os,re,math,numpy as np, collections.abc
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from pathlib import Path

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
                print(f"Computing interfacial tension at T = 0.9*{Tc_final}")
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

    def REFPROP_MIXTURE(self,mixture: str,z: list[float],T: float,pressures_bar,
                        kij: float = 0.0, phi_ij=1.0):
        
        """
        Compute mixture Interfacial tension over one or more pressures (bar).

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

        Returns
        -------
        pressures_bar : list[float]
        gamma_mN_per_m : list[float]
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

        # print(f"Preparing pure-component data at T={T} K")
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

        # --- sweep pressures
        results_gamma_Parachor = []
        results_gamma_WSD      = []
                
        for Pbar in pressures_bar:
            print(f"\nCalculating mixture gamma at T={T} K, P={Pbar} bar for {mixture}")

            mix = self.RP.REFPROPdll(mixture, "TP", "D;Dliq;Dvap", self.SI_BASE, 0, 1, T, Pbar/10.0, z)
            
            xL  = np.array(mix.x[:ncomp])
            yV  = np.array(mix.y[:ncomp])

            rhoL_kg_m3 = mix.Output[1]
            rhoV_kg_m3 = mix.Output[2]

            # phase molar masses (kg/kmol == g/mol numerically)
            MM_L = float(np.sum(xL * MOLAR_MASSES))
            MM_V = float(np.sum(yV * MOLAR_MASSES))

            # kg/m3 -> (kg/kmol) -> kmol/m3 -> mol/cm3
            rhoL_mol_cm3 = (rhoL_kg_m3 / MM_L) * 1e-3
            rhoV_mol_cm3 = (rhoV_kg_m3 / MM_V) * 1e-3

            gamma_mix_Parachor = self.compute_gamma_Parachor(
                                    x=xL.tolist(),
                                    y=yV.tolist(),
                                    rho_l=rhoL_mol_cm3,
                                    rho_v=rhoV_mol_cm3,
                                    parachor_numbers=parachor_numbers,
                                    kij=kij)
            
            print(f"Interfacial tension: {gamma_mix_Parachor:.6f} mN/m")
            results_gamma_Parachor.append(gamma_mix_Parachor)
            
            gamma_mix_WSD = self.compute_gamma_WSD(
                                    T=T, comp = components,
                                    x=xL.tolist(),
                                    y=yV.tolist(),
                                    rho_l=rhoL_mol_cm3,
                                    rho_v=rhoV_mol_cm3,
                                    phi=phi_ij)
            
            print(f"Interfacial tension: {gamma_mix_WSD:.6f} mN/m")
            results_gamma_WSD.append(gamma_mix_WSD)

        return pressures_bar, results_gamma_Parachor, results_gamma_WSD

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
        
    