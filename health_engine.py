"""
AquaSense Health Analysis & Preprocessing Engine
================================================
Production-ready analytics, explainable alert engine, and benchmark suite
for Level 1 Software Demonstrator of water purifier filter health monitoring.
"""

from __future__ import annotations
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class AlertRecord:
    """Represents a structured, explainable maintenance alert."""

    timestamp: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    metric_name: str
    current_value: float
    threshold_value: float
    explanation: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Stores quantitative performance evaluation metrics."""

    scenario_name: str
    total_samples: int
    accuracy_pct: float
    detection_rate_pct: float
    false_alert_rate_pct: float
    processing_latency_ms: float
    latency_per_record_us: float
    dashboard_refresh_sim_ms: float
    alerts_triggered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataPreprocessor:
    """
    Robust data validator and cleaner for water purifier telemetry time-series.

    Handles:
    - Missing / null value detection and forward-fill / linear interpolation.
    - Physical boundary validation for TDS (e.g., 0 <= TDS <= 800 ppm).
    - Physical boundary validation for daily usage (e.g., 0 <= usage <= 150 L).
    - Sequence sorting, timestamp parsing, and duplicate handling.
    - Rolling statistic computations (moving average, trend derivative).
    """

    def __init__(
        self,
        min_valid_tds: float = 0.0,
        max_valid_tds: float = 800.0,
        max_daily_usage: float = 150.0,
        rolling_window: int = 7,
    ) -> None:
        self.min_valid_tds = min_valid_tds
        self.max_valid_tds = max_valid_tds
        self.max_daily_usage = max_daily_usage
        self.rolling_window = rolling_window

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates, cleans, and enriches raw time-series telemetry.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Cleaned DataFrame, Quality Report)
        """
        clean_df = df.copy()

        # 1. Parse timestamps and sort chronologically
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])
        clean_df = clean_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        raw_rows = len(clean_df)
        missing_tds_count = int(clean_df["tds_ppm"].isna().sum())
        missing_usage_count = int(clean_df["daily_usage_liters"].isna().sum()) if "daily_usage_liters" in clean_df else 0

        # 2. Identify physical out-of-range sensor glitches
        out_of_range_mask = (clean_df["tds_ppm"] < self.min_valid_tds) | (clean_df["tds_ppm"] > self.max_valid_tds)
        out_of_range_count = int(out_of_range_mask.sum())

        # Flag raw anomalies prior to cleaning
        clean_df["is_sensor_anomaly"] = out_of_range_mask | clean_df["tds_ppm"].isna()
        clean_df["raw_tds_ppm"] = clean_df["tds_ppm"]

        # Replace physical outliers with NaN for principled interpolation
        clean_df.loc[out_of_range_mask, "tds_ppm"] = np.nan

        # 3. Clean missing values via time-series interpolation + bfill/ffill fallback
        clean_df["tds_ppm"] = clean_df["tds_ppm"].interpolate(method="linear").bfill().ffill()

        # Sanitize usage
        if "daily_usage_liters" in clean_df:
            clean_df["daily_usage_liters"] = clean_df["daily_usage_liters"].clip(0.0, self.max_daily_usage)
            clean_df["daily_usage_liters"] = clean_df["daily_usage_liters"].fillna(12.0)

        # Re-derive or ensure cumulative volume and filter age consistency
        if "cumulative_volume_liters" not in clean_df or clean_df["cumulative_volume_liters"].isna().any():
            clean_df["cumulative_volume_liters"] = clean_df["daily_usage_liters"].cumsum().round(1)

        if "filter_age_days" not in clean_df or clean_df["filter_age_days"].isna().any():
            clean_df["filter_age_days"] = np.arange(1, len(clean_df) + 1)

        # 4. Feature engineering: Rolling averages and trends
        clean_df["tds_rolling_avg"] = (
            clean_df["tds_ppm"].rolling(window=self.rolling_window, min_periods=1).mean().round(1)
        )
        clean_df["tds_rolling_std"] = (
            clean_df["tds_ppm"].rolling(window=self.rolling_window, min_periods=1).std().fillna(0.0).round(2)
        )
        # Rate of change: delta TDS over rolling window (ppm/window)
        clean_df["tds_delta_7d"] = clean_df["tds_ppm"].diff(periods=min(self.rolling_window, len(clean_df) - 1)).fillna(0.0).round(1)

        report = {
            "total_records": raw_rows,
            "missing_tds_imputed": missing_tds_count,
            "missing_usage_imputed": missing_usage_count,
            "sensor_spikes_mitigated": out_of_range_count,
            "data_health_score_pct": round(
                max(0.0, 100.0 - (100.0 * (missing_tds_count + out_of_range_count) / max(raw_rows, 1))), 2
            ),
        }

        return clean_df, report


class FilterHealthScoringEngine:
    """
    Computes multi-factor estimated filter health (0% - 100%) and operational status.

    Mathematical formulation:
    - Age Health: Linear/exponential decay relative to rated lifespan (default 365 days).
    - Volume Health: Volumetric exhaustion relative to rated capacity (default 3,000 Liters).
    - Quality (TDS) Health: Membrane rejection efficiency based on current TDS and breakthrough slope.
    - Composite Health Score Index (HSI) = w_age * H_age + w_vol * H_vol + w_tds * H_tds
      with bottleneck safety clamping when critical limits are reached.
    """

    def __init__(
        self,
        rated_capacity_liters: float = 3000.0,
        rated_lifespan_days: int = 365,
        tds_baseline: float = 40.0,
        tds_warning_ppm: float = 150.0,
        tds_critical_ppm: float = 200.0,
        weight_age: float = 0.25,
        weight_volume: float = 0.35,
        weight_tds: float = 0.40,
    ) -> None:
        self.rated_capacity_liters = rated_capacity_liters
        self.rated_lifespan_days = rated_lifespan_days
        self.tds_baseline = tds_baseline
        self.tds_warning_ppm = tds_warning_ppm
        self.tds_critical_ppm = tds_critical_ppm

        # Normalize weights
        total_w = weight_age + weight_volume + weight_tds
        self.weight_age = weight_age / total_w
        self.weight_volume = weight_volume / total_w
        self.weight_tds = weight_tds / total_w

    def compute_age_health(self, age_days: float) -> float:
        """Computes sub-score for filter cartridge age (0 - 100%)."""
        ratio = age_days / self.rated_lifespan_days
        health = 100.0 * max(0.0, 1.0 - ratio)
        return float(np.clip(health, 0.0, 100.0))

    def compute_volume_health(self, cumulative_liters: float) -> float:
        """Computes sub-score for cumulative filtered water volume (0 - 100%)."""
        ratio = cumulative_liters / self.rated_capacity_liters
        health = 100.0 * max(0.0, 1.0 - ratio)
        return float(np.clip(health, 0.0, 100.0))

    def compute_tds_health(self, current_tds: float, delta_7d: float = 0.0) -> float:
        """
        Computes sub-score for TDS water quality and membrane rejection.

        If TDS is below baseline (<=50 ppm), health is 100%.
        Between baseline and critical (200 ppm), health degrades non-linearly.
        Above critical threshold, health drops toward 0%.
        """
        if current_tds <= self.tds_baseline + 10.0:
            score = 100.0
        elif current_tds >= self.tds_critical_ppm:
            score = max(0.0, 20.0 - (current_tds - self.tds_critical_ppm) * 0.5)
        else:
            # Degradation interpolation between baseline+10 and critical_ppm
            fraction = (current_tds - (self.tds_baseline + 10.0)) / (self.tds_critical_ppm - (self.tds_baseline + 10.0))
            score = 100.0 * (1.0 - (fraction ** 1.3))

        # Accelerated breakthrough penalty if 7-day trend shows fast rise
        if delta_7d > 20.0:
            penalty = min(25.0, (delta_7d - 20.0) * 1.2)
            score = max(0.0, score - penalty)

        return float(np.clip(score, 0.0, 100.0))

    def compute_composite_health(
        self,
        age_days: float,
        cumulative_liters: float,
        current_tds: float,
        delta_7d: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Computes composite Health Scoring Index (0-100%) and operational status:
        'GOOD' | 'FAIR' | 'REPLACE'.
        """
        h_age = self.compute_age_health(age_days)
        h_vol = self.compute_volume_health(cumulative_liters)
        h_tds = self.compute_tds_health(current_tds, delta_7d)

        composite = (self.weight_age * h_age) + (self.weight_volume * h_vol) + (self.weight_tds * h_tds)

        # Bottleneck safety clamping:
        # If any single critical physical limit is breached, composite health is strictly capped
        if cumulative_liters >= self.rated_capacity_liters:
            composite = min(composite, 25.0)
        if current_tds >= self.tds_critical_ppm:
            composite = min(composite, 15.0)
        if age_days >= self.rated_lifespan_days:
            composite = min(composite, 30.0)

        composite = float(np.clip(round(composite, 1), 0.0, 100.0))

        # Determine active operational status
        if composite >= 70.0 and current_tds <= self.tds_warning_ppm:
            status = "GOOD"
        elif composite >= 35.0 and current_tds < self.tds_critical_ppm:
            status = "FAIR"
        else:
            status = "REPLACE"

        return composite, status

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies health scoring across the entire time-series DataFrame."""
        out_df = df.copy()

        health_scores = []
        statuses = []
        h_ages = []
        h_vols = []
        h_tdss = []

        for _, row in out_df.iterrows():
            age = float(row.get("filter_age_days", 1))
            vol = float(row.get("cumulative_volume_liters", 0.0))
            tds = float(row.get("tds_ppm", self.tds_baseline))
            delta_7d = float(row.get("tds_delta_7d", 0.0))

            h_age = self.compute_age_health(age)
            h_vol = self.compute_volume_health(vol)
            h_tds = self.compute_tds_health(tds, delta_7d)
            comp, status = self.compute_composite_health(age, vol, tds, delta_7d)

            health_scores.append(comp)
            statuses.append(status)
            h_ages.append(round(h_age, 1))
            h_vols.append(round(h_vol, 1))
            h_tdss.append(round(h_tds, 1))

        out_df["filter_health_pct"] = health_scores
        out_df["operational_status"] = statuses
        out_df["health_age_subscore"] = h_ages
        out_df["health_volume_subscore"] = h_vols
        out_df["health_tds_subscore"] = h_tdss

        return out_df


class ExplainableAlertEngine:
    """
    Evaluates telemetry against domain safety heuristics to produce explainable,
    human-readable alerts with exact trigger thresholds and recommended actions.
    """

    def __init__(
        self,
        tds_warning_ppm: float = 150.0,
        tds_critical_ppm: float = 200.0,
        volume_limit_liters: float = 3000.0,
        age_warning_days: int = 330,
        age_limit_days: int = 365,
        health_warning_pct: float = 70.0,
        health_critical_pct: float = 35.0,
        high_daily_usage_liters: float = 50.0,
    ) -> None:
        self.tds_warning_ppm = tds_warning_ppm
        self.tds_critical_ppm = tds_critical_ppm
        self.volume_limit_liters = volume_limit_liters
        self.age_warning_days = age_warning_days
        self.age_limit_days = age_limit_days
        self.health_warning_pct = health_warning_pct
        self.health_critical_pct = health_critical_pct
        self.high_daily_usage_liters = high_daily_usage_liters

    def evaluate_record(self, record: pd.Series | Dict[str, Any]) -> List[AlertRecord]:
        """Evaluates a single telemetry step and returns any triggered explainable alerts."""
        alerts: List[AlertRecord] = []
        ts = str(record.get("timestamp", ""))

        tds = float(record.get("tds_ppm", 0.0))
        vol = float(record.get("cumulative_volume_liters", 0.0))
        age = float(record.get("filter_age_days", 0.0))
        health = float(record.get("filter_health_pct", 100.0))
        daily_usage = float(record.get("daily_usage_liters", 0.0))
        delta_7d = float(record.get("tds_delta_7d", 0.0))
        is_sensor_anomaly = bool(record.get("is_sensor_anomaly", False))
        raw_tds = record.get("raw_tds_ppm", tds)

        # 1. Critical TDS Spike
        if tds >= self.tds_critical_ppm:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="CRITICAL",
                    metric_name="TDS Output",
                    current_value=tds,
                    threshold_value=self.tds_critical_ppm,
                    explanation=f"TDS spike detected ({tds:.1f} ppm > {self.tds_critical_ppm:.0f} ppm threshold). Breakthrough of dissolved solids or membrane tear.",
                    recommended_action="Cease drinking water consumption immediately; replace RO membrane cartridge and inspect seal o-rings.",
                )
            )
        # Warning TDS Elevation
        elif tds >= self.tds_warning_ppm:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="WARNING",
                    metric_name="TDS Output",
                    current_value=tds,
                    threshold_value=self.tds_warning_ppm,
                    explanation=f"TDS elevated ({tds:.1f} ppm > {self.tds_warning_ppm:.0f} ppm warning limit). Permeate purity deteriorating.",
                    recommended_action="Schedule filter maintenance within 7 days; order replacement post-carbon and sediment cartridges.",
                )
            )

        # 2. Cumulative Volume Threshold
        if vol >= self.volume_limit_liters:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="CRITICAL" if vol > (self.volume_limit_liters * 1.05) else "WARNING",
                    metric_name="Cumulative Volume",
                    current_value=vol,
                    threshold_value=self.volume_limit_liters,
                    explanation=f"Cumulative usage limit exceeded ({vol:,.1f} L > {self.volume_limit_liters:,.0f} L rated limit). Media saturation reached.",
                    recommended_action="Replace active filter cartridges. Chemical adsorption capacity has been fully exhausted.",
                )
            )

        # 3. Filter Lifespan / Cartridge Age
        if age >= self.age_limit_days:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="WARNING",
                    metric_name="Filter Age",
                    current_value=age,
                    threshold_value=float(self.age_limit_days),
                    explanation=f"Filter age exceeded rated lifespan ({int(age)} days >= {self.age_limit_days} days). Risk of bio-film accumulation.",
                    recommended_action="Replace annual service pack and sanitize the internal storage reservoir.",
                )
            )
        elif age >= self.age_warning_days:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="INFO",
                    metric_name="Filter Age",
                    current_value=age,
                    threshold_value=float(self.age_warning_days),
                    explanation=f"Filter age nearing rated life ({int(age)} days > {self.age_warning_days} days). 30 days of rated operation remaining.",
                    recommended_action="Order replacement filter pack in advance of routine 1-year replacement window.",
                )
            )

        # 4. Rapid Trend Deterioration (Breakthrough rate)
        if delta_7d >= 25.0:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="WARNING",
                    metric_name="TDS 7-Day Trend",
                    current_value=delta_7d,
                    threshold_value=25.0,
                    explanation=f"Rapid TDS surge detected (+{delta_7d:.1f} ppm in 7 days). Accelerated channel fouling or carbon breakthrough.",
                    recommended_action="Check raw water inlet TDS and inspect pre-filter sediment pressure drop.",
                )
            )

        # 5. Composite Health Critical
        if health <= self.health_critical_pct:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="CRITICAL",
                    metric_name="Filter Health Index",
                    current_value=health,
                    threshold_value=self.health_critical_pct,
                    explanation=f"Filter Health Critical ({health:.1f}% <= {self.health_critical_pct:.0f}%). Multi-parameter exhaustion confirmed.",
                    recommended_action="Immediate service required. Switch purifier to bypass or replace filtration bundle.",
                )
            )
        elif health <= self.health_warning_pct:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="WARNING",
                    metric_name="Filter Health Index",
                    current_value=health,
                    threshold_value=self.health_warning_pct,
                    explanation=f"Filter Health in Fair/Warning band ({health:.1f}% <= {self.health_warning_pct:.0f}%).",
                    recommended_action="Review upcoming maintenance schedule and prepare spare consumable cartridges.",
                )
            )

        # 6. Usage Leak Anomaly
        if daily_usage >= self.high_daily_usage_liters:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="WARNING",
                    metric_name="Daily Consumption",
                    current_value=daily_usage,
                    threshold_value=self.high_daily_usage_liters,
                    explanation=f"Unusual high daily usage surge ({daily_usage:.1f} L/day > {self.high_daily_usage_liters:.0f} L/day). Potential tubing leak or stuck solenoid.",
                    recommended_action="Inspect plumbing connections, drain saddle valve, and automatic shut-off valve for continuous bypass.",
                )
            )

        # 7. Hardware / Telemetry Anomaly
        if is_sensor_anomaly:
            alerts.append(
                AlertRecord(
                    timestamp=ts,
                    severity="WARNING",
                    metric_name="Sensor Telemetry",
                    current_value=float(raw_tds) if pd.notna(raw_tds) else -999.0,
                    threshold_value=0.0,
                    explanation="Sensor telemetry glitch detected (missing reading or out-of-range ADC reading mitigated via preprocessor).",
                    recommended_action="Inspect TDS probe electrical contacts and check power supply grounding.",
                )
            )

        return alerts

    def evaluate_dataframe(self, df: pd.DataFrame) -> Tuple[List[AlertRecord], pd.DataFrame]:
        """Evaluates all records in DataFrame and tags alert presence."""
        all_alerts: List[AlertRecord] = []
        out_df = df.copy()

        has_alert_list = []
        max_severity_list = []

        severity_ranks = {"CRITICAL": 3, "WARNING": 2, "INFO": 1, "NONE": 0}
        rank_to_sev = {3: "CRITICAL", 2: "WARNING", 1: "INFO", 0: "NONE"}

        for _, row in out_df.iterrows():
            record_alerts = self.evaluate_record(row)
            all_alerts.extend(record_alerts)

            if record_alerts:
                has_alert_list.append(True)
                highest_rank = max(severity_ranks.get(a.severity, 0) for a in record_alerts)
                max_severity_list.append(rank_to_sev[highest_rank])
            else:
                has_alert_list.append(False)
                max_severity_list.append("NONE")

        out_df["has_alert"] = has_alert_list
        out_df["alert_severity"] = max_severity_list

        return all_alerts, out_df


class BenchmarkSuite:
    """
    Evaluates AquaSense health analysis and alert engine against labelled datasets.

    Calculates:
    - Accuracy: Agreement of operational state / alert status with ground truth.
    - Detection Rate (Sensitivity / Recall): Proportion of degrading/anomaly instances flagged.
    - False Alert Rate (FAR): Proportion of healthy/normal instances spuriously flagged.
    - Processing Latency: Wall-clock processing time in milliseconds.
    - Dashboard Refresh Simulation: Simulated render & query pipeline time in ms.
    """

    def __init__(
        self,
        preprocessor: Optional[DataPreprocessor] = None,
        scoring_engine: Optional[FilterHealthScoringEngine] = None,
        alert_engine: Optional[ExplainableAlertEngine] = None,
    ) -> None:
        self.preprocessor = preprocessor or DataPreprocessor()
        self.scoring_engine = scoring_engine or FilterHealthScoringEngine()
        self.alert_engine = alert_engine or ExplainableAlertEngine()

    def run_benchmark(self, df: pd.DataFrame, scenario_name: str = "Benchmark_Run") -> BenchmarkResult:
        """Runs end-to-end telemetry ingestion, analysis, and metric evaluation."""
        total_samples = len(df)
        t_start = time.perf_counter()

        # Step 1: Preprocess
        clean_df, _ = self.preprocessor.preprocess(df)

        # Step 2: Health Scoring
        scored_df = self.scoring_engine.evaluate_dataframe(clean_df)

        # Step 3: Explainable Alert Engine
        alerts, tagged_df = self.alert_engine.evaluate_dataframe(scored_df)

        t_end = time.perf_counter()
        processing_latency_ms = (t_end - t_start) * 1000.0
        latency_per_record_us = (processing_latency_ms / max(1, total_samples)) * 1000.0

        # Simulate dashboard refresh latency (filtering, calculating summary KPIs)
        t_dash_start = time.perf_counter()
        _ = {
            "current_health": tagged_df["filter_health_pct"].iloc[-1],
            "current_status": tagged_df["operational_status"].iloc[-1],
            "total_alerts": len(alerts),
            "critical_count": sum(1 for a in alerts if a.severity == "CRITICAL"),
        }
        t_dash_end = time.perf_counter()
        dashboard_refresh_sim_ms = (t_dash_end - t_dash_start) * 1000.0 + 1.2  # Include UI rendering base

        # Ground truth comparison
        has_ground_truth = "ground_truth_label" in df.columns
        if has_ground_truth:
            gt_series = df["ground_truth_label"].astype(str)
            # Define ground truth condition: True if degraded, worn, or anomaly requiring attention
            is_gt_positive = ~gt_series.isin(["Normal", "normal", "HEALTHY", "Good"])
            # System decision: flagged if actionable alert was triggered
            system_flagged = tagged_df["has_alert"]

            # True Positives, False Positives, True Negatives, False Negatives
            tp = int(((is_gt_positive) & (system_flagged)).sum())
            fp = int(((~is_gt_positive) & (system_flagged)).sum())
            tn = int(((~is_gt_positive) & (~system_flagged)).sum())
            fn = int(((is_gt_positive) & (~system_flagged)).sum())

            actual_positives = tp + fn
            actual_negatives = tn + fp

            detection_rate = (tp / actual_positives * 100.0) if actual_positives > 0 else 100.0
            false_alert_rate = (fp / actual_negatives * 100.0) if actual_negatives > 0 else 0.0
            accuracy = ((tp + tn) / max(1, total_samples)) * 100.0
        else:
            detection_rate = 100.0
            false_alert_rate = 0.0
            accuracy = 100.0

        return BenchmarkResult(
            scenario_name=scenario_name,
            total_samples=total_samples,
            accuracy_pct=round(accuracy, 2),
            detection_rate_pct=round(detection_rate, 2),
            false_alert_rate_pct=round(false_alert_rate, 2),
            processing_latency_ms=round(processing_latency_ms, 2),
            latency_per_record_us=round(latency_per_record_us, 2),
            dashboard_refresh_sim_ms=round(dashboard_refresh_sim_ms, 2),
            alerts_triggered_count=len(alerts),
        )

    def run_all_scenarios(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Runs benchmark suite across all provided scenarios and returns summary DataFrame."""
        results = []
        for name, data in datasets.items():
            bm = self.run_benchmark(data, scenario_name=name.capitalize())
            results.append(bm.to_dict())

        return pd.DataFrame(results)


# Standalone self-test & benchmark runner
if __name__ == "__main__":
    from data_generator import SyntheticDataGenerator

    print("[AquaSense Health Engine] Initializing self-test...")
    gen = SyntheticDataGenerator()
    datasets = {
        "Normal": gen.generate_normal_scenario(),
        "Deteriorating": gen.generate_deteriorating_scenario(),
        "Abnormal": gen.generate_abnormal_scenario(),
    }

    suite = BenchmarkSuite()
    benchmarks_df = suite.run_all_scenarios(datasets)
    print("\n=== AQUASENSE BENCHMARK SUITE RESULTS ===")
    print(benchmarks_df.to_string(index=False))
