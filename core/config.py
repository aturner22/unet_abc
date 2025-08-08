import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


class Config:
    def __init__(self, config_path: str | Path, timestamp: str | None = None):
        with open(config_path, "r") as f:
            raw = json.load(f)

        self.config_path = str(config_path)
        self.raw: dict[str, Any] = raw

        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self.name: str = raw["name"] + "_" + timestamp
        self.sample_size: int = raw["sample_size"]
        self.ensemble_size: int = raw["ensemble_size"]
        self.n_gibbs_steps: int = raw["n_gibbs_steps"]
        self.n_proposals_per_variable: int = raw["n_proposals_per_variable"]
        self.proposal_scale: float = raw["proposal_scale"]
        self.temporal_resampling: bool = raw.get("temporal_resampling", False)
        self.score_function: str = raw["score_function"]

        self.inference_mode: str = raw["inference_mode"]
        self.abc_epsilon: int = raw["abc_epsilon"]
        self.adaptive_epsilon: bool = raw["adaptive_epsilon"]
        self.min_epsilon: float = raw["min_epsilon"]
        self.epsilon_quantile: float = raw["epsilon_quantile"]

        abc_params = raw.get("abc_params", {})
        self.initial_alpha_range: tuple = tuple(
            abc_params.get("initial_alpha_range", [0.05, 1.5])
        )
        self.min_alpha: float = abc_params.get("min_alpha", 1e-4)
        self.adapt_every: int = abc_params.get("adapt_every", 5)
        self.adapt_factor: float = abc_params.get("adapt_factor", 0.85)
        self.adapt_stop: int = abc_params.get("adapt_stop", 100)
        self.eps_energy: float = abc_params.get("eps_energy", 1e-6)
        self.checkpoint_file: str = abc_params.get(
            "checkpoint_file", "gibbs_checkpoint_step.npz"
        )

        scoring_params = raw.get("scoring_params", {})
        self.ensemble_threshold_exact: int = scoring_params.get(
            "ensemble_threshold_exact", 10
        )
        self.sampling_approximation_size: int = scoring_params.get(
            "sampling_approximation_size", 500
        )

        self.memory_management = raw.get("memory_management", {})

        self.variable_names: list[str] = raw["variable_names"]
        self.num_variables: int = len(self.variable_names)
        self.num_static_fields: int = raw["num_static_fields"]
        self.max_horizon: int = raw["max_horizon"]
        self.spacing: int = raw["spacing"]
        self.t_max: int = raw["t_max"]
        self.t_direct: int = raw["t_direct"]
        self.t_iter: int = raw["t_iter"]
        self.conditioning_times: list[int] = raw.get("conditioning_times", [0])
        self.n_ens: int = raw["n_ens"]
        self.SEED: int = raw["seed"]

        self.data_directory: Path = Path(raw["data_directory"])
        self.model_directory: Path = Path(raw["model_directory"])
        self.result_directory: Path = Path(raw["result_directory"]) / self.name
        self.result_directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        return self.raw


def load_config(config_path: str | Path) -> Config:
    return Config(config_path)
