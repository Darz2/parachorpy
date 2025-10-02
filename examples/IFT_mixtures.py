#!/usr/bin/env python
# Created by Darshan on 2025-08-15

from parachorpy import parachor as IFT
import numpy as np, csv

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

    # Pbar, gamma_Parachor, gamma_WSD = model.REFPROP_MIXTURE(MIXTURE, z, T, PRESSURES, kij, phi_ij)
    
    # for P, s1, s2  in zip(Pbar, gamma_Parachor, gamma_WSD):
    #     print(f"P = {P:.1f} bar | gamma_Parachor = {s1:.4f} mN/m | gamma_WSD = {s2:.4f} mN/m")
        
    # with open("IFT_results.csv", "w", newline="") as csvfile:
        
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Pressure [bar]", "gamma_Parachor [mN/m]", "gamma_WSD [mN/m]"])
        
    #     for P, s1, s2 in zip(Pbar, gamma_Parachor, gamma_WSD):
    #         writer.writerow([f"{P:.1f}", f"{s1:.4f}", f"{s2:.4f}"])
            
    pxy = model.REFPROP_PXY(MIXTURE, T, npts=50, clip=1e-3, verbose=False)

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
