#!/usr/bin/env python
# Created by Darshan on 2025-10-03

from parachorpy import parachor as IFT
import numpy as np, pandas as pd, csv
import matplotlib.pyplot as plt

def make_branch_df(branch, P, x, y, rhoLkg, rhoVkg, rhoLmol, rhoVmol, Mliq, Mvap):
    P = np.asarray(P, dtype=float)
    rhoLkg = np.asarray(rhoLkg, dtype=float)
    rhoVkg = np.asarray(rhoVkg, dtype=float)
    rhoLmol = np.asarray(rhoLmol, dtype=float)
    rhoVmol = np.asarray(rhoVmol, dtype=float)
    Mliq = np.asarray(Mliq, dtype=float)
    Mvap = np.asarray(Mvap, dtype=float)

    mask = np.isfinite(P) & np.isfinite(rhoLkg) & np.isfinite(rhoVkg) & \
    np.isfinite(rhoLmol) & np.isfinite(rhoVmol) & \
    np.isfinite(Mliq) & np.isfinite(Mvap)

    return pd.DataFrame({
    "Branch": branch,
    "Pressure [bar]": P[mask],
    "x_CO2 [-]": [xi[0] for i, xi in enumerate(x) if mask[i]],
    "x_Ar [-]": [xi[1] for i, xi in enumerate(x) if mask[i]],
    "y_CO2 [-]": [yi[0] for i, yi in enumerate(y) if mask[i]],
    "y_Ar [-]": [yi[1] for i, yi in enumerate(y) if mask[i]],
    "rho_L [kg/m3]": rhoLkg[mask],
    "rho_V [kg/m3]": rhoVkg[mask],
    "rho_L_mol_cm3 [mol/cm3]": rhoLmol[mask],
    "rho_V_mol_cm3 [mol/cm3]": rhoVmol[mask],
    "M_liq [kg/mol]": Mliq[mask],
    "M_vap [kg/mol]": Mvap[mask],
    })

if __name__ == '__main__':
    
    json_files  = ["Cachadina_2015.json","Mulero_2012.json"]
    # json_files  = ["Mulero_2012.json"]
    model       = IFT.InterfacialTension(json_files)
    model.ctREFPROP_init('~/Software/REFPROP/REFPROP-cmake/build', gerg_enable=1)

    MIXTURE    = "CO2;Nitrogen"
    components = MIXTURE.split(";")
    T          = 220.0  # K
    kij        = 0.0
    phi_ij     = 1.0
            
    pxy = model.RP_PXY(MIXTURE, T, npts=1000, clip=1e-3, verbose=False, return_densities=True)
    
    # # --- Bubble branch ---
    P_bub               = pxy['bubble']['P_bar']
    x_bub               = pxy['bubble']['x']
    y_bub               = pxy['bubble']['y']
    rhoL_bub_kg_m3      = pxy['bubble']['rhoL_kg_m3']
    rhoV_bub_kg_m3      = pxy['bubble']['rhoV_kg_m3']
    rhoL_bub_mol_cm3    = pxy['bubble']['rhoL_mol_cm3']
    rhoV_bub_mol_cm3    = pxy['bubble']['rhoV_mol_cm3']
    Mliq_bub            = pxy['bubble']['M_liq']
    Mvap_bub            = pxy['bubble']['M_vap']

    # --- Dew branch ---
    P_dew               = pxy['dew']['P_bar']
    x_dew               = pxy['dew']['x']
    y_dew               = pxy['dew']['y']
    rhoL_dew_kg_m3      = pxy['dew']['rhoL_kg_m3']
    rhoV_dew_kg_m3      = pxy['dew']['rhoV_kg_m3']
    rhoL_dew_mol_cm3    = pxy['dew']['rhoL_mol_cm3']
    rhoV_dew_mol_cm3    = pxy['dew']['rhoV_mol_cm3']
    Mliq_dew            = pxy['dew']['M_liq']
    Mvap_dew            = pxy['dew']['M_vap']


    df_bubble = make_branch_df("Bubble", P_bub, x_bub, y_bub,
    rhoL_bub_kg_m3, rhoV_bub_kg_m3,
    rhoL_bub_mol_cm3, rhoV_bub_mol_cm3,
    Mliq_bub, Mvap_bub)

    df_dew = make_branch_df("Dew", P_dew, x_dew, y_dew,
    rhoL_dew_kg_m3, rhoV_dew_kg_m3,
    rhoL_dew_mol_cm3, rhoV_dew_mol_cm3,
    Mliq_dew, Mvap_dew)
    
    df_all = pd.concat([df_bubble, df_dew], ignore_index=True)
    df_all.sort_values(by="Pressure [bar]", inplace=True)
    df_all.to_csv(rf"../DATA/CO2-N2/Pxy_{components[0]}-{components[1]}_{T:.0f}.csv", index=False, encoding="utf-8")

# _____________________ Interfacial Tension Calculations _____________________ #

    gamma_bub = model.gamma_pxy(MIXTURE, T, P_bub, x_bub, y_bub, rhoL_bub_mol_cm3, rhoV_bub_mol_cm3,
                                kij, phi_ij, verbose=True, enforce_monotone=True)
    
    P_bub             = gamma_bub["P_bar"]
    gamma_Parachor    = gamma_bub['gamma_Parachor_mNpm']
    gamma_WSD         = gamma_bub['gamma_WSD_mNpm']
    WSD_mixcorr       = gamma_bub['mixcorr_WSD']
    x_bub             = gamma_bub['x']
    y_bub             = gamma_bub['y']
    
    df_ift = pd.DataFrame({
                "P [bar]": P_bub,
                "x_CO2 [-]": [xi[0] for xi in x_bub],
                "y_CO2 [-]": [yi[0] for yi in y_bub],
                "gamma_Parachor [mN/m]": gamma_Parachor,
                "gamma_WSD [mN/m]": gamma_WSD,
                "mixcorr_WSD": WSD_mixcorr
                }).sort_values(by="P [bar]")
    df_ift.to_csv(rf"../DATA/CO2-N2/IFT_P_{components[0]}-{components[1]}_{T:.0f}.csv", index=False, encoding="utf-8")
            
# dft                = pd.read_csv('IFT_binary/pcsaft_CO2_Ar.csv')
# dft_kij            = pd.read_csv('IFT_binary/pcsaft_CO2_Ar_kij.csv')
# # Sort the DataFrame by Pressure before plotting
# dft_sorted = dft.sort_values(by='Pressure[bar]')
# dft_kij_sorted = dft_kij.sort_values(by='Pressure[bar]')
# # plt.plot(dft_sorted['Pressure[bar]'], dft_sorted['surface_tension[mN/m]'], '-', color='k', label='PC-SAFT + cDFT, kij = 0')
# # plt.plot(dft_kij_sorted['Pressure[bar]'], dft_kij_sorted['surface_tension[mN/m]'], '--', color='k', label='PC-SAFT + cDFT, kij = 0.0541')
# plt.plot(P_bub, gamma_Parachor, label="Parachor")
# plt.plot(P_bub, gamma_WSD, label="WSD")
# plt.xlabel("Pressure [bar]")
# plt.ylabel("Interfacial Tension [mN/m]")
# plt.title(f"Interfacial Tension vs Pressure for {MIXTURE} at T={T} K")
# plt.legend()
# plt.grid()
# plt.savefig(rf"../DATA/CO2-N2/IFT_P_{components[0]}-{components[1]}_{T:.0f}.png", dpi=300)