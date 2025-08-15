#!/usr/bin/env python

# Created by Darshan on 2025-07-15

import json,os,re,numpy as np, collections.abc
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from pathlib import Path

class SurfaceTension:
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

    def compute_sigma_pure(self, T, fluid_name, Tc=None):
        """
        Compute surface tension for a fluid at temperature T.

        Parameters:
        - T : float
        - fluid_name : str
        - Tc : float (optional) — use only if Tc is not in the JSON

        Returns:
        - sigma : float (mN/m)
        """
        Tc_json, s, n = self.get_fluid_params(fluid_name)
        Tc_final = Tc_json if Tc_json is not None else Tc

        if Tc_final is None:
            raise ValueError(f"No Tc provided in JSON or as input for {fluid_name}")

        if len(s) != len(n):
            raise ValueError("Length of s and n must be equal")

        if T <= Tc_final:
            sigma_T = sum(si * (1 - T / Tc_final)**ni for si, ni in zip(s, n))

        if T > Tc_final:
            print(f"Given temperature is greater than Tc ({Tc_final}) for {fluid_name}")
            print(f"Computing surface tension at T = 0.9*{Tc_final}")
            T = 0.9 * Tc_final
            sigma_T = sum(si * (1 - T / Tc_final)**ni for si, ni in zip(s, n))
            
            return sigma_T * 1e3, T  # Convert to mN/m

        return sigma_T * 1e3, T  # Convert to mN/m

    def parachor_number(self,rho_l, rho_v, sigma):
        """
        Calculate the parachor number.

        Parameters:
        - rho_l : float — liquid density [kg/m³]
        - rho_v : float — vapor density [kg/m³]
        - sigma : float — surface tension [mN/m]

        Returns:
        - P : float — parachor number [kg/m³]ⁿ·mN/m

        Constant:
        - n : float — exponent in the parachor equation (default: 3.87); REFPROP v10 (Lemmon et al, 2018)
        """
        n = 3.87
        if rho_l <= 0 or rho_v <= 0 or sigma <= 0:
            raise ValueError("All inputs must be positive")

        delta_rho = rho_l - rho_v               # [mol/cm³]
        sigma_SI = sigma                        # in  mN/m
        parachor = sigma_SI**(1/n) / delta_rho
        
        return parachor

    def REFPROP_MIXTURE(
        self,
        mixture: str,
        z: list[float],
        T: float,
        pressures_bar,
        kij: float = 0.0,
    ):
        """
        Compute mixture surface tension over one or more pressures (bar).

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
        sigma_mN_per_m : list[float]
        """

        # Allow single float or iterable
        if isinstance(pressures_bar, collections.abc.Iterable) and not isinstance(pressures_bar, (str, bytes)):
            pressures_bar = list(pressures_bar)
        else:
            pressures_bar = [pressures_bar]
        
        components = mixture.split(";")
        ncomp = len(components)
        pressures_bar = list(pressures_bar)

        # --- per-component molar masses & parachor numbers at Teff (from pure sigma)
        MOLAR_MASSES = []
        parachor_numbers = []

        # print(f"Preparing pure-component data at T={T} K")
        for comp in components:
            # Molar mass (kg/kmol == g/mol numerically)
            MM = self.RP.REFPROPdll(comp, "TQ", "M", self.SI_BASE, 0, 0, T, 0, [1.0]).Output[0]
            MOLAR_MASSES.append(MM)

            # Pure sigma (may switch to Teff=0.9Tc if T>Tc)
            sigma_val_mNpm, Teff = self.compute_sigma_pure(T, comp, Tc=None)

            liq = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 0, [1.0])
            vap = self.RP.REFPROPdll(comp, "TQ", "D;P", self.SI_BASE, 0, 0, Teff, 1, [1.0])

            rhoL_kg_m3 = liq.Output[0]
            rhoV_kg_m3 = vap.Output[0]

            # Convert densities to mol/cm^3 and compute parachor number
            P_i = self.parachor_number(
                rhoL_kg_m3 / (MM * 1e3),
                rhoV_kg_m3 / (MM * 1e3),
                sigma_val_mNpm
            )
            parachor_numbers.append(P_i)

        # --- sweep pressures
        results_sigma = []
        for Pbar in pressures_bar:
            print(f"\nCalculating mixture sigma at T={T} K, P={Pbar} bar for {mixture}")

            mix = self.RP.REFPROPdll(mixture, "TP", "D;Dliq;Dvap", self.SI_BASE, 0, 1, T, Pbar/10.0, z)
            xL = np.array(mix.x[:ncomp])
            yV = np.array(mix.y[:ncomp])

            rhoL_kg_m3 = mix.Output[1]
            rhoV_kg_m3 = mix.Output[2]

            # phase molar masses (kg/kmol == g/mol numerically)
            MM_L = float(np.sum(xL * MOLAR_MASSES))
            MM_V = float(np.sum(yV * MOLAR_MASSES))

            # kg/m3 -> (kg/kmol) -> kmol/m3 -> mol/cm3
            rhoL_mol_cm3 = (rhoL_kg_m3 / MM_L) * 1e-3
            rhoV_mol_cm3 = (rhoV_kg_m3 / MM_V) * 1e-3

            sigma_mix = self.compute_sigma_mixture(
                x=xL.tolist(),
                y=yV.tolist(),
                rho_l=rhoL_mol_cm3,
                rho_v=rhoV_mol_cm3,
                parachor_numbers=parachor_numbers,
                kij=kij
            )
            print(f"Surface tension: {sigma_mix:.6f} mN/m")
            results_sigma.append(sigma_mix)

        return pressures_bar, results_sigma

    def compute_sigma_mixture(self, x, y, rho_l, rho_v, parachor_numbers, kij=0.0):
        """
        Compute surface tension for a mixture using the parachor method.

        Parameters:
        - x, y : mole fractions (lists) for liquid and vapor
        - rho_l, rho_v : densities [mol/cm^3]
        - parachor_numbers : list of component parachors (P_i)
        - kij : scalar or matrix of binary interaction parameters (default 0)

        Returns:
        - sigma_mix : surface tension [mN/m]
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

        sigma_mix = (rho_l * P_l - rho_v * P_v) ** n
        
        return sigma_mix