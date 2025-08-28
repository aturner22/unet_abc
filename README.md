# Score-ABC-RFP: Approximate Bayesian Computation with Proper Scoring Rules

This repository implements a high-dimensional Approximate Bayesian Computation framework using Gibbs sampling with rejection-based forward proposals (ABC-RFP) using proper scoring rules for spatiotemporal model calibration.

Adapted from [Continuous Ensemble Weather Forecasting with Diffusion Models](https://arxiv.org/abs/2410.05431). We thank Martin Andrae for providing a pretrained U-Net used in this repository.

## Methodological Summary

* ABC-Gibbs inference over perturbation coefficients $\alpha \in \mathbb{R}^P_+$.
* Proposals generated via scaled differences of ERA5 field pairs.
* Forecast ensemble generated via pretrained deterministic U-Net.
* Discrepancy minimisation via proper scoring (CRPS / Energy).
* Adaptive epsilon scheduling and memory-aware batch evaluation.

## Data Preparation

Download 5.625° ERA5 data from [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895):

```
era5_data/
   |-- 10m_u_component_of_wind
   |-- 10m_v_component_of_wind
   |-- 2m_temperature
   |-- constants
   |-- geopotential_500
   |-- temperature_850
```

Run `create_dataset.py` to convert the data into `.npy` format required by the code.

## Execution

To run ABC-RFP inference:

```bash
python main.py --config path/to/config.json
```

For HPC environments:

```bash
export CONFIG_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
```
