#!/usr/bin/env python
# Created by Darshan on 2025-08-15

from Interfacethermopy import parachor as IFT
import numpy as np, pandas as pd, csv

if __name__ == '__main__':
    
    json_files  = ["Mulero_2012.json"]
    model       = IFT.InterfacialTension(json_files)
    model.ctREFPROP_init('~/Software/REFPROP_BETA/REFPROP-cmake/build', gerg_enable=1)

    MIXTURE   = "CO2;Methane"
    z         = [0.95, 0.05]
    T         = 250.0
    PRESSURES = np.arange(20.0, 31.0, 0.5)
    kij       = 0.0
    phi_ij    = 1.0

    Pbar, gamma_Parachor, gamma_WSD = model.gamma_TP(MIXTURE, z, T, PRESSURES, kij, phi_ij)
    
    # for P, s1, s2  in zip(Pbar, gamma_Parachor, gamma_WSD):
    #     print(f"P = {P:.1f} bar | gamma_Parachor = {s1:.4f} mN/m | gamma_WSD = {s2:.4f} mN/m")
        
    # with open("IFT_results.csv", "w", newline="") as csvfile:
        
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Pressure [bar]", "gamma_Parachor [mN/m]", "gamma_WSD [mN/m]"])
        
    #     for P, s1, s2 in zip(Pbar, gamma_Parachor, gamma_WSD):
    #         writer.writerow([f"{P:.1f}", f"{s1:.4f}", f"{s2:.4f}"])
            
    pxy = model.RP_PXY(MIXTURE, T, npts=50, clip=1e-3, verbose=False, return_densities=True)
    
    # --- Bubble branch ---
    P_bub    = pxy['bubble']['P_bar']
    x_bub    = pxy['bubble']['x']
    y_bub    = pxy['bubble']['y']
    rhoL_bub = pxy['bubble']['rhoL']
    rhoV_bub = pxy['bubble']['rhoV']
    Mliq_bub = pxy['bubble']['M_liq']
    Mvap_bub = pxy['bubble']['M_vap']

    # --- Dew branch ---
    P_dew    = pxy['dew']['P_bar']
    x_dew    = pxy['dew']['x']
    y_dew    = pxy['dew']['y']
    rhoL_dew = pxy['dew']['rhoL']
    rhoV_dew = pxy['dew']['rhoV']
    Mliq_dew = pxy['dew']['M_liq']
    Mvap_dew = pxy['dew']['M_vap']
    
    # print("\n--- Bubble curve ---")
    # for x, y, P, rhoL, rhoV, M_liq, M_vap in zip(
    #     pxy['bubble']['x'], 
    #     pxy['bubble']['y'], 
    #     pxy['bubble']['P_bar'], 
    #     pxy['bubble']['rhoL'], 
    #     pxy['bubble']['rhoV'],
    #     pxy['bubble']['M_liq'],
    #     pxy['bubble']['M_vap']
    # ):
    #     if not np.isnan(P):
    #         print(
    #             f"x_CO2={x[0]:.4f}, y_CO2={y[0]:.4f}, "
    #             f"P={P:.3f} bar, rhoL={rhoL:.3f} kg/m³, rhoV={rhoV:.3f} kg/m³, "
    #             f"M_liq={M_liq:.6f} kg/mol, M_vap={M_vap:.6f} kg/mol"
    #         )

    # print("\n--- Dew curve ---")
    # for y, x, P, rhoL, rhoV, M_liq, M_vap in zip(
    #     pxy['dew']['y'],
    #     pxy['dew']['x'],
    #     pxy['dew']['P_bar'],
    #     pxy['dew']['rhoL'],
    #     pxy['dew']['rhoV'],
    #     pxy['dew']['M_liq'],
    #     pxy['dew']['M_vap']
    # ):
    #     if np.isfinite(P):
    #         print(
    #             f"y_CO2={y[0]:.4f}, x_CO2={x[0]:.4f}, "
    #             f"P={P:.3f} bar, rhoL={rhoL:.3f} kg/m³, rhoV={rhoV:.3f} kg/m³, "
    #             f"M_liq={M_liq:.6f} kg/mol, M_vap={M_vap:.6f} kg/mol"
    #         )

    
    # --- Build DataFrame ---
    df_bubble = pd.DataFrame({
        "branch": "bubble",
        "P_bar": P_bub,
        "x_CO2": [xi[0] for xi in x_bub],
        "y_CO2": [yi[0] for yi in y_bub],
        "rhoL_kgm3": rhoL_bub,
        "rhoV_kgm3": rhoV_bub,
        "Mliq_kgmol": Mliq_bub,
        "Mvap_kgmol": Mvap_bub,
    })

    df_dew = pd.DataFrame({
        "branch": "dew",
        "P_bar": P_dew,
        "x_CO2": [xi[0] for xi in x_dew],
        "y_CO2": [yi[0] for yi in y_dew],
        "rhoL_kgm3": rhoL_dew,
        "rhoV_kgm3": rhoV_dew,
        "Mliq_kgmol": Mliq_dew,
        "Mvap_kgmol": Mvap_dew,
    })

    # Combine and save
    df_all = pd.concat([df_bubble, df_dew], ignore_index=True)

    df_all.to_csv("pxy_output.csv", index=False)

    # gamma_bub = model.gamma_pxy(MIXTURE, T, P_bub, x_bub, y_bub, rhoL_bub, rhoV_bub, Mliq_bub, Mvap_bub, kij, phi_ij, verbose=False)