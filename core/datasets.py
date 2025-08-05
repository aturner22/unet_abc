import numpy as np
import torch
from torch.utils.data import Sampler


class ERA5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_mode,
        sample_counts,
        dimensions,
        lead_time,
        max_horizon,
        norm_factors,
        device,
        lead_time_range,
        spinup=0,
        spacing=1,
        dtype="float32",
        conditioning_times=[0, -6],
        static_data_path=None,
        random_lead_time=0,
    ):
        self.dataset_path = dataset_path
        self.data_dtype = dtype
        self.device = device

        self.dataset_mode = dataset_mode
        self.n_samples, self.n_train, self.n_val = sample_counts
        self.num_variables, self.n_lat, self.n_lon = dimensions
        self.max_horizon = max_horizon
        self.lead_time = lead_time
        self.spinup = spinup + 24
        self.spacing = spacing
        self.mean, self.std_dev = norm_factors
        self.t_min, self.t_max, self.delta_t = lead_time_range

        self.static_data_path = static_data_path
        self.static_fields = None
        self.static_vars = 0

        if static_data_path is not None:
            self.static_fields = self.load_static_data()
            self.static_vars = self.static_fields.shape[0]

        self.conditioning_times = conditioning_times
        self.random_lead_time = random_lead_time

        if dataset_mode == "train":
            self.start_index = self.spinup
            self.end_index = self.n_train
        elif dataset_mode == "val":
            self.start_index = self.n_train
            self.end_index = self.n_train + self.n_val
        else:
            self.start_index = self.n_train + self.n_val
            self.end_index = self.n_samples

        self.indices = list(range(self.start_index, self.end_index, self.spacing))

        shape = (self.n_samples, self.num_variables, self.n_lat, self.n_lon)
        self.data = np.memmap(
            self.dataset_path, dtype=self.data_dtype, mode="r", shape=shape
        )

    def load_static_data(self):
        static_shape = (2, self.n_lat, self.n_lon)
        static_data = np.memmap(
            self.static_data_path, dtype=self.data_dtype, mode="r", shape=static_shape
        )
        mins = static_data.min(axis=(1, 2), keepdims=True)
        maxs = static_data.max(axis=(1, 2), keepdims=True)
        rng = np.maximum(maxs - mins, 1.0)
        scaled = (static_data - mins) / rng

        return torch.as_tensor(scaled, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]

        if self.random_lead_time:
            lead_time = np.random.randint(self.t_min, self.t_max + 1)
        else:
            lead_time = (
                self.lead_time[0]
                if isinstance(self.lead_time, list)
                else self.lead_time
            )

        if data_idx + lead_time >= self.n_samples:
            data_idx = self.n_samples - lead_time - 1

        input_fields = []
        for t in self.conditioning_times:
            if data_idx + t >= 0 and data_idx + t < self.n_samples:
                field = torch.tensor(
                    self.data[data_idx + t], dtype=torch.float32, device=self.device
                )
                input_fields.append(field)

        if self.static_fields is not None:
            input_fields.append(self.static_fields)

        if self.static_fields is not None:
            self.static_fields = torch.nan_to_num(
                self.static_fields, nan=0.0, posinf=0.0, neginf=0.0
            )

        input_tensor = torch.cat(input_fields, dim=0)
        target_tensor = torch.tensor(
            self.data[data_idx + lead_time], dtype=torch.float32, device=self.device
        )

        n_actual_fields = len(input_fields) - (
            1 if self.static_fields is not None else 0
        )
        n_met_channels = n_actual_fields * self.num_variables

        if self.static_fields is not None and input_tensor.shape[0] > n_met_channels:
            met_vars = input_tensor[:n_met_channels]
            static_vars = input_tensor[n_met_channels:]

            expanded_mean = torch.tensor(self.mean).repeat(n_actual_fields).reshape(-1, 1, 1).to(input_tensor.device)
            expanded_std = torch.tensor(self.std_dev).repeat(n_actual_fields).reshape(-1, 1, 1).to(input_tensor.device)

            met_vars_norm = (met_vars - expanded_mean) / expanded_std
            input_tensor = torch.cat([met_vars_norm, static_vars], dim=0)
        else:
            expanded_mean = torch.tensor(self.mean).repeat(n_actual_fields).reshape(-1, 1, 1).to(input_tensor.device)
            expanded_std = torch.tensor(self.std_dev).repeat(n_actual_fields).reshape(-1, 1, 1).to(input_tensor.device)
            input_tensor = (
                input_tensor[:n_met_channels] - expanded_mean
            ) / expanded_std

        target_tensor = (
            target_tensor - torch.tensor(self.mean).reshape(-1, 1, 1).to(target_tensor.device)
        ) / torch.tensor(self.std_dev).reshape(-1, 1, 1).to(target_tensor.device)

        return input_tensor, target_tensor, lead_time


class FixedSampler(Sampler):
    def __init__(self, data_source, sample_size):
        self.data_source = data_source
        self.sample_size = min(sample_size, len(data_source))

    def __iter__(self):
        return iter(range(self.sample_size))

    def __len__(self):
        return self.sample_size
