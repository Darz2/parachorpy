# parachorpy

## Overview

`parachorpy` is a Python package that calculates **surface tension of pure fluids** using correlations [1,2] and **surface tension of mixtures** using parachor model [3] as a function of temperature and pressure. It supports loading JSON data sources containing fluid-specific coefficients.

## Features

- Read one or more JSON files with surface tension correlation parameters.
- Retrieve fluid-specific parameters (![s](https://latex.codecogs.com/svg.latex?s_i), ![n](https://latex.codecogs.com/svg.latex?n_i), and optionally ![Tc](https://latex.codecogs.com/svg.latex?T_c)).
- Compute surface tension of pure fluids ![sigma](https://latex.codecogs.com/svg.latex?\sigma(T)) using the following correlation:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\sigma(T)=\sum_i%20s_i\left(1-\frac{T}{T_c}\right)^{n_i}" alt="\sigma(T) = \sum_i s_i (1 - T/T_c)^{n_i}">
</p>


- Compute the surface tension of mixtures using Parachor method using the following correlation:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\gamma_{\mathrm{mix}}=\left(\rho_{l}\sum_{j=1}^{N}\sum_{i=1}^{N}x_i%20x_j%20\mathcal{P}_{ij}-\rho_{v}\sum_{j=1}^{N}\sum_{i=1}^{N}y_i%20y_j%20\mathcal{P}_{ij}\right)^n" alt="\gamma_{\mathrm{mix}} = (\rho_l \sum_{j=1}^N \sum_{i=1}^N x_i x_j \mathcal{P}_{ij} - \rho_v \sum_{j=1}^N \sum_{i=1}^N y_i y_j \mathcal{P}_{ij})^n">
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathcal{P}_{ij}=(1-\delta_{ij})\frac{\mathcal{P}_{i}+\mathcal{P}_{j}}{2}" alt="\mathcal{P}_{ij} = (1 - \delta_{ij}) \frac{\mathcal{P}_{i} + \mathcal{P}_{j}}{2}">
</p>

- ![delta_ij](https://latex.codecogs.com/svg.latex?\delta_{ij}) is the fit parameter (binary interaction parameter).
- The component-specific parachor number ![P_i](https://latex.codecogs.com/svg.latex?\mathcal{P}_{i}) is obtained at the given temperature of the mixture ![T_mix](https://latex.codecogs.com/svg.latex?T_{mix}).


<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\sigma^{1/n}=\mathcal{P}_{i}(\rho_{l}-\rho_{v})" alt="\sigma^{1/n} = \mathcal{P}_{i} (\rho_l - \rho_v)">
</p>

- ![n](https://latex.codecogs.com/svg.latex?n) = 3.87 and ![P_i_T](https://latex.codecogs.com/svg.latex?\mathcal{P}_{i}(T)) is the Parachor number of the fluid.
- If ![T_mix](https://latex.codecogs.com/svg.latex?T_{mix}) > 0.9*![T_c](https://latex.codecogs.com/svg.latex?T_{c}) i.e., the critical temperature of a component in the mixture, then ![P_i](https://latex.codecogs.com/svg.latex?\mathcal{P}_{i}) is computed at 0.9*![T_c](https://latex.codecogs.com/svg.latex?T_{c}) for numerical stability. This approach is also used in REFPROP V10.
- The Parachor model is used in addition with other Equation of State (EoS) models (Peng–Robinson, SRK, GERG-2008, EoS-CG, etc.) to compute ![rho_l](https://latex.codecogs.com/svg.latex?\rho_l), ![rho_v](https://latex.codecogs.com/svg.latex?\rho_v), ![P_T](https://latex.codecogs.com/svg.latex?\mathcal{P}(T)), ![x](https://latex.codecogs.com/svg.latex?x), and ![y](https://latex.codecogs.com/svg.latex?y).
- This allows the Parachor model to be used in conjunction with other openly available thermodynamic packages like Clapeyron, FeOs, REFPROP, and similar other thermodynamic packages.


## Installation

No external packages are required beyond Python's standard libraries.

```bash
python3 example/IFT_mixtures.py
```

## JSON Input Format

Each JSON file must contain a list of dictionaries, each describing one fluid. Example:

```json
[
  {
    "Fluid": "methane",
    "Tc": 190.56,
    "s": [0.2358, 0.0121],
    "n": [1.23, 2.34]
  },
  {
    "Fluid": "argon",
    "Tc": 150.86,
    "s": [0.152],
    "n": [1.26]
  }
]
```

## Example Usage

```python
from parachorpy import parachor as ST
import numpy as np

if __name__ == '__main__':
    json_files = ["Mulero_2012.json"]
    model = ST.SurfaceTension(json_files)
    model.ctREFPROP_init('~/Software/REFPROP_BETA/REFPROP-cmake/build', gerg_enable=1)

    MIXTURE   = "CO2;Methane"
    z         = [0.95, 0.05] # composition
    T         = 250.0        # in Kelvin
    PRESSURES = np.arange(20.0, 31.0, 5.0) # in bar
    kij       = 0.0

    Pbar, sigma = model.REFPROP_MIXTURE(MIXTURE, z, T, PRESSURES, kij)

    for P, s in zip(Pbar, sigma):
        print(f"P = {P:.1f} bar | sigma = {s:.4f} mN/m")
```

## References

1. **Cachadina, I., Vega, L. F., & de Miguel, E. (2015)**  
   Empirical correlations for the surface tension of pure fluids  
   *Journal of Chemical Thermodynamics*, 87, 162–170.  
   [https://doi.org/10.1063/1.4921749](https://doi.org/10.1063/1.4921749)  

2. **Mulero, A., Cachadina, I., & Parra, M. I. (2012)**  
   Recommended correlations for the surface tension of common fluids  
   *Journal of Physical and Chemical Reference Data*, 41(4), 043105.  
   [https://doi.org/10.1063/5.0277723](https://doi.org/10.1063/5.0277723)  

3. **Sugden, S. (1924)**  
   A relation between surface tension, density, and chemical composition  
   *Journal of the Chemical Society, Transactions*, 125, 32–41.  
   [https://doi.org/10.1039/CT9242500032](https://doi.org/10.1039/CT9242500032)  

> **Note:** More correlations can be added for n-alkanes, ethers, and esters.
