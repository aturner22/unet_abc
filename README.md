# ABC-RFP: Approximate Bayesian Computation with Rejection-based Forward Proposals

This repository implements a high-dimensional Approximate Bayesian Computation framework using Gibbs sampling with rejection-based forward proposals (ABC-RFP) for spatiotemporal model calibration. The methodology is non-parametric and ensemble-based, evaluating discrepancy between generated and reference atmospheric states via proper scoring rules (CRPS, Energy Score).

Adapted from [Continuous Ensemble Weather Forecasting with Diffusion Models](https://arxiv.org/abs/2410.05431). We thank Martin Andrae for providing a pretrained U-Net used in this repository.

## Methodological Summary

* ABC-Gibbs inference over perturbation coefficients $\alpha \in \mathbb{R}^P_+$.
* Proposals generated via scaled differences of ERA5 field pairs.
* Forecast ensemble generated via pretrained deterministic U-Net.
* Discrepancy minimisation via proper scoring (CRPS / Energy).
* Adaptive epsilon scheduling and memory-aware batch evaluation.

## Prerequisites

Tested with Python 3.10+ and PyTorch 2.0+.

**Dependencies:**

* `pytorch`, `numpy`, `pandas`, `tqdm`, `matplotlib`
* `zarr`, `xarray`, `jupyter`, `ipykernel`
* `cartopy` (for plotting),

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

## Output

Posterior statistics and diagnostics are saved to the configured result directory:

* `posterior_samples.npy`, `posterior_scores.npy`
* `posterior_mean.npy`, `posterior_variance.npy`
* `results.npz`

## License

Research use only. Contact authors for commercial licensing or reuse inquiries.
