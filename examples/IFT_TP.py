#!/usr/bin/env python
# Created by Darshan on 2025-10-02

from parachorpy import parachor as IFT
import numpy as np, pandas as pd, csv

if __name__ == '__main__':
    
    json_files  = ["Mulero_2012.json"]
    model       = IFT.InterfacialTension(json_files)
    model.ctREFPROP_init('~/Software/REFPROP_BETA/REFPROP-cmake/build', gerg_enable=1)

    MIXTURE   = "CO2;Methane"
    z         = [0.95, 0.05]
    T         = 250.0
    PRESSURES = np.arange(15, 31.0, 0.5)
    kij       = 0.0
    phi_ij    = 1.0

    Pbar, gamma_Parachor, gamma_WSD = model.gamma_TP(MIXTURE, z, T, PRESSURES, kij, phi_ij)
    
    for P, s1, s2  in zip(Pbar, gamma_Parachor, gamma_WSD):
        print(f"P = {P:.1f} bar | gamma_Parachor = {s1:.4f} mN/m | gamma_WSD = {s2:.4f} mN/m")
        
    with open("IFT_results.csv", "w", newline="") as csvfile:
        
        writer = csv.writer(csvfile)
        writer.writerow(["Pressure [bar]", "gamma_Parachor [mN/m]", "gamma_WSD [mN/m]"])
        
        for P, s1, s2 in zip(Pbar, gamma_Parachor, gamma_WSD):
            writer.writerow([f"{P:.1f}", f"{s1:.4f}", f"{s2:.4f}"])
            