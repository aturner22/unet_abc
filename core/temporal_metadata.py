import datetime
import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TemporalMetadata:
    def __init__(self, start_year: int = 1979, end_year: int = 2018):
        self.start_year = start_year
        self.end_year = end_year
        
        self.timestamps = pd.date_range(
            datetime.datetime(start_year, 1, 1, 0),
            datetime.datetime(end_year, 12, 31, 23),
            freq="1h"
        )
        
        self._create_temporal_indices()
        
    def _create_temporal_indices(self):
        self.day_of_year = np.array([ts.dayofyear for ts in self.timestamps])
        
        self.hour_of_day = np.array([ts.hour for ts in self.timestamps])
        
        self.year = np.array([ts.year for ts in self.timestamps])
        
        self.month = np.array([ts.month for ts in self.timestamps])
        
        self.season = np.array([self._get_season(ts.month) for ts in self.timestamps])
        
        logger.info(f"Created temporal metadata for {len(self.timestamps)} time steps")
        logger.info(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        
    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "DJF" 
        elif month in [3, 4, 5]:
            return "MAM" 
        elif month in [6, 7, 8]:
            return "JJA"
        else: 
            return "SON"  
    
    def get_seasonal_diurnal_candidates(
        self,
        base_idx: int,
        day_window: int = 30,
        hour_tolerance: int = 0,
        exclude_same_year: bool = True
    ) -> np.ndarray:

        if base_idx >= len(self.timestamps):
            raise ValueError(f"Base index {base_idx} exceeds dataset size {len(self.timestamps)}")
            
        base_day = self.day_of_year[base_idx]
        base_hour = self.hour_of_day[base_idx]
        base_year = self.year[base_idx]
        
        day_candidates = self._get_day_candidates(base_day, day_window)
        hour_candidates = self._get_hour_candidates(base_hour, hour_tolerance)
        valid_indices = np.intersect1d(day_candidates, hour_candidates)
        
        if exclude_same_year:
            different_year_mask = self.year[valid_indices] != base_year
            valid_indices = valid_indices[different_year_mask]
        
        return valid_indices
    
    def _get_day_candidates(self, base_day: int, day_window: int) -> np.ndarray:
        if base_day <= day_window:
            day_mask = (
                (self.day_of_year >= base_day - day_window) |
                (self.day_of_year <= base_day + day_window) |
                (self.day_of_year >= 365 - (day_window - base_day))
            )
        elif base_day >= 365 - day_window:
            day_mask = (
                (self.day_of_year >= base_day - day_window) |
                (self.day_of_year <= base_day + day_window) |
                (self.day_of_year <= (base_day + day_window) - 365)
            )
        else:
            day_mask = (
                (self.day_of_year >= base_day - day_window) &
                (self.day_of_year <= base_day + day_window)
            )
        
        return np.where(day_mask)[0]
    
    def _get_hour_candidates(self, base_hour: int, hour_tolerance: int) -> np.ndarray:
        if hour_tolerance == 0:
            return np.where(self.hour_of_day == base_hour)[0]
        
        hour_min = (base_hour - hour_tolerance) % 24
        hour_max = (base_hour + hour_tolerance) % 24
        
        if hour_min <= hour_max:
            hour_mask = (
                (self.hour_of_day >= hour_min) & 
                (self.hour_of_day <= hour_max)
            )
        else:
            hour_mask = (
                (self.hour_of_day >= hour_min) | 
                (self.hour_of_day <= hour_max)
            )
        
        return np.where(hour_mask)[0]
    
    def get_statistics_for_candidates(
        self, 
        base_idx: int, 
        candidates: np.ndarray
    ) -> Dict[str, any]:
        """Get statistical information about candidate indices."""
        if len(candidates) == 0:
            return {
                'count': 0,
                'years_covered': [],
                'seasons_covered': [],
                'day_range': (None, None),
                'hour_range': (None, None)
            }
        
        return {
            'count': len(candidates),
            'years_covered': sorted(list(set(self.year[candidates]))),
            'seasons_covered': sorted(list(set(self.season[candidates]))),
            'day_range': (self.day_of_year[candidates].min(), self.day_of_year[candidates].max()),
            'hour_range': (self.hour_of_day[candidates].min(), self.hour_of_day[candidates].max()),
            'base_timestamp': str(self.timestamps[base_idx]),
            'candidate_sample_timestamps': [str(self.timestamps[idx]) for idx in candidates[:5]] 
        }


def create_temporal_metadata(
    start_year: int = 1979,
    end_year: int = 2018
) -> TemporalMetadata:
    return TemporalMetadata(start_year, end_year)
