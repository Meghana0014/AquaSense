"""
AquaSense Synthetic Data Generator
==================================
Generates realistic, time-series synthetic datasets for water purifier filter
health monitoring and predictive maintenance demonstration.

Scenarios Covered:
1. Normal Operation: Stable TDS, expected cumulative volume progression, no premature failure.
2. Deteriorating Filter: Gradual membrane fouling and carbon block exhaustion over usage.
3. Abnormal / Anomaly: Acute membrane rupture, sensor glitches, intermittent spikes, missing values.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """
    Production-grade synthetic time-series generator for water purifier telemetry.

    Columns produced:
    - timestamp: ISO 8601 formatted date/time string.
    - tds_ppm: Total Dissolved Solids in parts per million (clean water output).
    - daily_usage_liters: Water volume filtered during the day.
    - cumulative_volume_liters: Running total of volume filtered across filter lifetime.
    - filter_age_days: Cumulative age of the filter cartridge in days.
    - ground_truth_label: Semantic classification label for benchmarking.
    """

    def __init__(
        self,
        base_start_date: str = "2025-01-01",
        rated_capacity_liters: float = 3000.0,
        rated_lifespan_days: int = 365,
        random_seed: int = 42,
    ) -> None:
        self.base_start_date = pd.to_datetime(base_start_date)
        self.rated_capacity_liters = rated_capacity_liters
        self.rated_lifespan_days = rated_lifespan_days
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def generate_normal_scenario(
        self,
        days: int = 180,
        avg_daily_usage: float = 8.0,
        base_tds: float = 35.0,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generates healthy baseline telemetry with mild stochastic variations.

        - TDS stays safely in the range of 30-55 ppm.
        - Usage follows realistic household consumption (6-12 L/day).
        - Cumulative volume stays well under rated capacity (3,000 L).
        """
        rng = np.random.default_rng(seed if seed is not None else self.random_seed)
        date_range = pd.date_range(start=self.base_start_date, periods=days, freq="D")

        # Daily usage with normal variation + weekend surge
        usage = rng.normal(loc=avg_daily_usage, scale=1.8, size=days)
        # Add slight weekend multiplier
        for i, dt in enumerate(date_range):
            if dt.weekday() >= 5:
                usage[i] *= 1.2
        usage = np.clip(usage, a_min=3.0, a_max=16.0).round(1)

        cumulative_volume = np.cumsum(usage).round(1)
        filter_age = np.arange(1, days + 1)

        # Baseline TDS with gentle seasonal drift + low white noise
        tds_drift = 5.0 * np.sin(np.linspace(0, 2 * np.pi, days))
        tds_noise = rng.normal(loc=0.0, scale=2.2, size=days)
        tds = (base_tds + tds_drift + tds_noise).round(1)
        tds = np.clip(tds, 20.0, 65.0)

        df = pd.DataFrame(
            {
                "timestamp": date_range.strftime("%Y-%m-%d %H:%M:%S"),
                "tds_ppm": tds,
                "daily_usage_liters": usage,
                "cumulative_volume_liters": cumulative_volume,
                "filter_age_days": filter_age,
                "ground_truth_label": "Normal",
            }
        )
        return df

    def generate_deteriorating_scenario(
        self,
        days: int = 280,
        avg_daily_usage: float = 14.5,
        base_tds: float = 40.0,
        degradation_start_day: int = 130,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generates progressive filter wear scenario modeling membrane fouling and carbon exhaustion.

        - Starts normal, but as cumulative volume approaches and surpasses ~3,000 L and
          filter age extends past 200+ days, mineral rejection efficiency drops.
        - TDS follows a progressive non-linear curve up to 180-220 ppm.
        - Labels shift: Normal -> Deteriorating -> Critical_Maintenance.
        """
        rng = np.random.default_rng(seed if seed is not None else self.random_seed + 101)
        date_range = pd.date_range(start=self.base_start_date, periods=days, freq="D")

        usage = rng.normal(loc=avg_daily_usage, scale=3.0, size=days)
        usage = np.clip(usage, a_min=6.0, a_max=32.0).round(1)
        cumulative_volume = np.cumsum(usage).round(1)
        filter_age = np.arange(1, days + 1)

        tds = np.zeros(days, dtype=float)
        labels = []

        for i in range(days):
            day_idx = i + 1
            vol = cumulative_volume[i]
            base_noise = rng.normal(0.0, 2.5)

            if day_idx < degradation_start_day:
                tds[i] = base_tds + base_noise
                labels.append("Normal")
            elif day_idx < 210:
                # Early fouling / wear
                progress = max(0.0, float(day_idx - degradation_start_day) / float(210 - degradation_start_day))
                tds_elevated = base_tds + (progress ** 1.6) * 55.0 + base_noise
                tds[i] = tds_elevated
                labels.append("Deteriorating")
            else:
                # Severe wear / breakthrough approaching or exceeding rated limits
                late_progress = max(0.0, float(day_idx - 210) / max(1.0, float(days - 210)))
                tds_severe = 95.0 + (late_progress ** 1.3) * 115.0 + base_noise
                tds[i] = tds_severe
                if tds_severe > 150.0 or vol > self.rated_capacity_liters:
                    labels.append("Critical_Maintenance")
                else:
                    labels.append("Deteriorating")

        tds = np.clip(tds, 25.0, 240.0).round(1)

        # Inject 1-2 realistic mild missing readings to simulate real telemetry drops
        missing_indices = rng.choice(np.arange(30, days - 10), size=3, replace=False)
        tds[missing_indices] = np.nan

        df = pd.DataFrame(
            {
                "timestamp": date_range.strftime("%Y-%m-%d %H:%M:%S"),
                "tds_ppm": tds,
                "daily_usage_liters": usage,
                "cumulative_volume_liters": cumulative_volume,
                "filter_age_days": filter_age,
                "ground_truth_label": labels,
            }
        )
        return df

    def generate_abnormal_scenario(
        self,
        days: int = 150,
        avg_daily_usage: float = 8.5,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generates acute mechanical and electrical fault telemetry with edge cases:
        - Sudden membrane rupture / seal bypass spike (e.g., jump from 45 ppm to 320 ppm).
        - Sensor glitch spikes (e.g., 999 ppm or negative values).
        - Missing values (sensor drops / disconnected ADC).
        - Unusually high usage burst (leak / stuck flush valve: 85 L/day).
        - Ground truth labels indicate Anomaly / Abnormal points explicitly.
        """
        rng = np.random.default_rng(seed if seed is not None else self.random_seed + 202)
        date_range = pd.date_range(start=self.base_start_date, periods=days, freq="D")

        usage = rng.normal(loc=avg_daily_usage, scale=1.8, size=days)
        usage = np.clip(usage, a_min=4.0, a_max=18.0).round(1)

        # Acute leak anomaly on day 55-58
        usage[55:59] = [68.0, 92.5, 84.0, 75.0]

        cumulative_volume = np.cumsum(usage).round(1)
        filter_age = np.arange(1, days + 1)

        tds = rng.normal(loc=42.0, scale=3.0, size=days)
        labels = ["Normal"] * days

        # 1. Leak anomaly labels
        for idx in range(55, 59):
            labels[idx] = "Abnormal_HighUsage"

        # 2. Sudden membrane seal failure starting at day 85
        rupture_duration = 18
        for j in range(85, min(days, 85 + rupture_duration)):
            tds[j] = 260.0 + rng.normal(0.0, 15.0) + (j - 85) * 5.0
            labels[j] = "Abnormal_MembraneRupture"

        # 3. Intermittent out-of-range sensor spikes (hardware glitch / open circuit)
        spike_indices = [22, 41, 115, 134]
        spike_values = [-12.5, 999.0, 850.0, -5.0]
        for s_idx, s_val in zip(spike_indices, spike_values):
            if s_idx < days:
                tds[s_idx] = s_val
                labels[s_idx] = "Sensor_Glitch_Outliers"

        # 4. Sensor telemetry dropouts / missing values (NaN)
        nan_indices = [15, 30, 72, 122]
        for n_idx in nan_indices:
            if n_idx < days:
                tds[n_idx] = np.nan
                labels[n_idx] = "Sensor_Data_Missing"

        tds = np.round(tds, 1)

        df = pd.DataFrame(
            {
                "timestamp": date_range.strftime("%Y-%m-%d %H:%M:%S"),
                "tds_ppm": tds,
                "daily_usage_liters": usage,
                "cumulative_volume_liters": cumulative_volume,
                "filter_age_days": filter_age,
                "ground_truth_label": labels,
            }
        )
        return df

    def generate_all_scenarios(self, output_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Generates and saves the three standard reference scenario datasets to CSV files.
        """
        target_path = Path(output_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        datasets = {
            "normal": (self.generate_normal_scenario(), "normal_scenario.csv"),
            "deteriorating": (self.generate_deteriorating_scenario(), "deteriorating_scenario.csv"),
            "abnormal": (self.generate_abnormal_scenario(), "abnormal_scenario.csv"),
        }

        saved_dfs = {}
        for key, (df, filename) in datasets.items():
            file_dest = target_path / filename
            df.to_csv(file_dest, index=False)
            saved_dfs[key] = df
            print(f"[AquaSense Generator] Exported: {file_dest} ({len(df)} rows)")

        return saved_dfs


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_all_scenarios(output_dir="data")
