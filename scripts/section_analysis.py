import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# =============================================================================
# STATE_FLOAT INDEX REFERENCE (from Linesight project)
# =============================================================================
# 0:        mini_race_time
# 1-20:     previous actions (5 actions × 4 inputs)
# 21-24:    wheel_sliding (4 wheels)
# 25-28:    wheel_ground_contact (4 wheels)
# 29-32:    wheel_damper_absorb (4 wheels)
# 33:       gearbox_state
# 34:       gear
# 35:       rpm
# 36:       gearbox_counter
# 37-52:    wheel_material (4 wheels × 4 material types)
# 53-55:    angular_velocity (x, y, z)
# 56-58:    linear_velocity (lateral, vertical, forward)
# 59-61:    y_map_vector
# 62-181:   zone_centers (40 VCPs × 3 coords)
# 182:      margin_to_finish
# 183:      is_freewheeling

class StateFloatSchema:
    """Indices for state_float array."""
    LATERAL_VEL = 56
    VERTICAL_VEL = 57
    FORWARD_VEL = 58
    WHEEL_SLIDING_START = 21
    WHEEL_SLIDING_END = 25
    WHEEL_GROUND_START = 25
    WHEEL_GROUND_END = 29
    WHEEL_DAMPER_START = 29
    WHEEL_DAMPER_END = 33
    ANGULAR_VEL_START = 53
    ANGULAR_VEL_END = 56
    IS_FREEWHEELING = 183


def load_data(data_dir=".", max_rows=50_000):
    """Load and filter rollout data."""
    print(f"Looking for parquet files in {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, "merged_rollouts_part_*.parquet")))
    
    if not files:
        if os.path.exists(os.path.join(data_dir, "merged_rollouts.parquet")):
            files = [os.path.join(data_dir, "merged_rollouts.parquet")]
        else:
            print("No merged_rollouts parquet files found.")
            return None

    print(f"Found {len(files)} files. Loading...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows.")
    
    # Keep recent data for freshness
    if len(df) > max_rows:
        print(f"Filtering to last {max_rows} rows...")
        df = df.iloc[-max_rows:].copy()
    
    # Clean data - remove rows with missing or inconsistent state_float
    df = df.dropna(subset=["state_float"])
    lengths = df["state_float"].apply(len)
    mode_len = lengths.mode()[0]
    df = df[lengths == mode_len].copy()
    
    return df


def extract_features(df):
    """Extract all relevant features from state_float."""
    print("Extracting features...")
    state_floats = np.vstack(df["state_float"].values)
    schema = StateFloatSchema
    
    # ==========================================================================
    # VELOCITY COMPONENTS
    # ==========================================================================
    lateral = state_floats[:, schema.LATERAL_VEL]
    vertical = state_floats[:, schema.VERTICAL_VEL]
    forward = state_floats[:, schema.FORWARD_VEL]
    
    df["lateral_speed"] = np.abs(lateral)
    df["vertical_speed"] = vertical
    df["forward_speed"] = forward
    df["speed_kmh"] = np.sqrt(lateral**2 + vertical**2 + forward**2) * 3.6
    
    # Drift intensity
    df["drift_ratio"] = df["lateral_speed"] / (np.abs(df["forward_speed"]) + 0.1)
    
    # ==========================================================================
    # WHEEL PHYSICS
    # ==========================================================================
    df["wheels_sliding"] = state_floats[:, schema.WHEEL_SLIDING_START:schema.WHEEL_SLIDING_END].sum(axis=1)
    df["wheels_on_ground"] = state_floats[:, schema.WHEEL_GROUND_START:schema.WHEEL_GROUND_END].sum(axis=1)
    df["avg_damper"] = state_floats[:, schema.WHEEL_DAMPER_START:schema.WHEEL_DAMPER_END].mean(axis=1)
    
    # Angular velocity
    df["yaw_rate"] = np.abs(state_floats[:, schema.ANGULAR_VEL_START + 1])
    
    # Freewheeling
    if state_floats.shape[1] > schema.IS_FREEWHEELING:
        df["is_freewheeling"] = state_floats[:, schema.IS_FREEWHEELING]
    
    # ==========================================================================
    # CRASH DETECTION (JUMP-SAFE + PROGRESSIVE START THRESHOLD)
    # ==========================================================================
    # IMPORTANT: Define 'zone' BEFORE using it in np.select()!
    # ==========================================================================
    
    zone = df["current_zone_idx"].values  # ← MUST BE HERE, BEFORE np.select()
    
    crash_threshold = np.select(
        condlist=[
            zone <= 30,       # Launch phase
            zone <= 100,      # Initial acceleration
            zone <= 200,      # Building speed
            zone <= 400,      # Approaching race pace
            zone <= 600,      # Transition to full speed
        ],
        choicelist=[
            0,                # No crash possible
            3,                # Very lenient
            5,                # Lenient
            7,                # Moderate
            9,                # Slightly lenient
        ],
        default=10            # Normal threshold (zone 601+)
    )
    
    # Store threshold for debugging
    df["crash_threshold"] = crash_threshold
    
    # Apply crash detection
    df["is_crash"] = (
        (df["speed_kmh"] < crash_threshold) |
        (df["forward_speed"] < -5)
    )
    
    # Additional diagnostic flags
    df["is_airborne"] = df["wheels_on_ground"] == 0
    df["is_backwards"] = df["forward_speed"] < -2
    df["is_heavy_drift"] = df["drift_ratio"] > 0.5
    
    return df


def analyze_zone_performance(df, output_dir):
    """Comprehensive zone-by-zone performance analysis."""
    print("--- Zone Performance Analysis ---")
    
    # Ensure required columns exist
    required_cols = ["current_zone_idx", "speed_kmh", "is_crash"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found.")
            return None
    
    # Aggregate statistics per zone
    zone_stats = df.groupby("current_zone_idx").agg(
        total_frames=("speed_kmh", "count"),
        avg_speed=("speed_kmh", "mean"),
        std_speed=("speed_kmh", "std"),
        crash_count=("is_crash", "sum"),
        airborne_count=("is_airborne", "sum"),
        backwards_count=("is_backwards", "sum"),
        heavy_drift_count=("is_heavy_drift", "sum"),
        avg_drift_ratio=("drift_ratio", "mean"),
        avg_wheels_ground=("wheels_on_ground", "mean"),
    ).reset_index()
    
    # Calculate rates
    zone_stats["crash_rate"] = (zone_stats["crash_count"] / zone_stats["total_frames"]) * 100
    zone_stats["airborne_rate"] = (zone_stats["airborne_count"] / zone_stats["total_frames"]) * 100
    zone_stats["backwards_rate"] = (zone_stats["backwards_count"] / zone_stats["total_frames"]) * 100
    
    # Fill NaN std with 0
    zone_stats["std_speed"] = zone_stats["std_speed"].fillna(0)
    
    # Identify kill zones (crash rate > 25% with sufficient samples)
    kill_zones = zone_stats[
        (zone_stats["crash_rate"] > 25) & 
        (zone_stats["total_frames"] > 20)
    ].sort_values("crash_rate", ascending=False)
    
    print(f"\n🔴 KILL ZONES DETECTED: {len(kill_zones)}")
    if len(kill_zones) > 0:
        print(kill_zones[["current_zone_idx", "crash_rate", "avg_speed", "total_frames"]].head(10))
    
    # Save results
    zone_stats.to_csv(os.path.join(output_dir, "zone_stats.csv"), index=False)
    kill_zones.to_csv(os.path.join(output_dir, "kill_zones.csv"), index=False)
    
    # Generate visualizations
    _plot_crash_rate(zone_stats, output_dir)
    _plot_speed_profile(zone_stats, output_dir)
    _plot_failure_modes(zone_stats, output_dir)
    
    return zone_stats


def analyze_zone_transitions(df, output_dir):
    """Analyze zone-to-zone transition failures (regression detection)."""
    print("--- Zone Transition Analysis ---")
    
    # Sort by rollout and frame if available
    if "rollout_id" in df.columns:
        df = df.sort_values(["rollout_id", "frame_idx"]).copy()
    else:
        df = df.copy()
    
    # Calculate next zone
    df["next_zone"] = df["current_zone_idx"].shift(-1)
    df["zone_regressed"] = df["next_zone"] < df["current_zone_idx"]
    
    # Filter to same rollout transitions (if rollout_id exists)
    if "rollout_id" in df.columns:
        df["same_rollout"] = df["rollout_id"] == df["rollout_id"].shift(-1)
        df = df[df["same_rollout"]]
    
    # Aggregate transition stats
    transitions = df.groupby("current_zone_idx").agg(
        transitions=("next_zone", "count"),
        regressions=("zone_regressed", "sum")
    ).reset_index()
    
    transitions["regression_rate"] = (transitions["regressions"] / transitions["transitions"]) * 100
    
    # High regression zones (where runs often fail)
    problem_zones = transitions[
        (transitions["regression_rate"] > 20) & 
        (transitions["transitions"] > 10)
    ].sort_values("regression_rate", ascending=False)
    
    print(f"\n⚠️ HIGH REGRESSION ZONES: {len(problem_zones)}")
    if len(problem_zones) > 0:
        print(problem_zones.head(10))
    
    transitions.to_csv(os.path.join(output_dir, "zone_transitions.csv"), index=False)
    
    return transitions


def _plot_crash_rate(zone_stats, output_dir):
    """Plot crash rate by zone (top 50 worst zones)."""
    # Filter for zones with significant traffic
    significant = zone_stats[zone_stats["total_frames"] > 10].copy()
    top_crash = significant.sort_values("crash_rate", ascending=False).head(50)
    
    if len(top_crash) == 0:
        print("No significant zones to plot for crash rate.")
        return
    
    plt.figure(figsize=(14, 8))
    colors = ["#d62728" if cr > 40 else "#ff7f0e" if cr > 25 else "#2ca02c" 
              for cr in top_crash["crash_rate"]]
    plt.bar(range(len(top_crash)), top_crash["crash_rate"], color=colors)
    plt.xticks(range(len(top_crash)), top_crash["current_zone_idx"].astype(int), rotation=90)
    plt.axhline(y=25, color='orange', linestyle='--', label='Warning (25%)')
    plt.axhline(y=40, color='red', linestyle='--', label='Critical (40%)')
    plt.title("Top 50 Struggle Zones by Crash Rate")
    plt.xlabel("Zone ID")
    plt.ylabel("Crash Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zone_crash_rate.png"), dpi=150)
    plt.close()


def _plot_speed_profile(zone_stats, output_dir):
    """Plot speed profile across track."""
    # Sort by zone for proper line plot
    sorted_stats = zone_stats.sort_values("current_zone_idx")
    
    plt.figure(figsize=(14, 6))
    plt.fill_between(sorted_stats["current_zone_idx"], 
                     sorted_stats["avg_speed"] - sorted_stats["std_speed"],
                     sorted_stats["avg_speed"] + sorted_stats["std_speed"],
                     alpha=0.3, label="±1 std")
    plt.plot(sorted_stats["current_zone_idx"], sorted_stats["avg_speed"], 
             color="blue", linewidth=2, label="Avg Speed")
    plt.title("Speed Profile Across Track")
    plt.xlabel("Zone ID")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speed_profile.png"), dpi=150)
    plt.close()


def _plot_failure_modes(zone_stats, output_dir):
    """Plot failure mode breakdown (crash, airborne, drift)."""
    # Filter for zones with sufficient data
    significant = zone_stats[zone_stats["total_frames"] > 50].copy()
    
    if len(significant) == 0:
        print("Not enough data for failure modes plot.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Crash rate
    scatter1 = axes[0].scatter(significant["current_zone_idx"], significant["crash_rate"], 
                               c=significant["crash_rate"], cmap="Reds", alpha=0.7, s=20)
    axes[0].set_title("Crash Rate by Zone")
    axes[0].set_xlabel("Zone ID")
    axes[0].set_ylabel("Crash Rate (%)")
    axes[0].axhline(y=25, color='orange', linestyle='--', alpha=0.5)
    
    # Airborne rate (informational - jumps are OK!)
    scatter2 = axes[1].scatter(significant["current_zone_idx"], significant["airborne_rate"],
                               c=significant["airborne_rate"], cmap="Blues", alpha=0.7, s=20)
    axes[1].set_title("Airborne Rate by Zone (Jumps OK)")
    axes[1].set_xlabel("Zone ID")
    axes[1].set_ylabel("Airborne Rate (%)")
    
    # Drift ratio
    scatter3 = axes[2].scatter(significant["current_zone_idx"], significant["avg_drift_ratio"],
                               c=significant["avg_drift_ratio"], cmap="Oranges", alpha=0.7, s=20)
    axes[2].set_title("Avg Drift Ratio by Zone")
    axes[2].set_xlabel("Zone ID")
    axes[2].set_ylabel("Drift Ratio")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "failure_modes.png"), dpi=150)
    plt.close()


def generate_report(zone_stats, output_dir):
    """Generate a human-readable summary report."""
    report = []
    report.append("=" * 60)
    report.append("SECTION ANALYSIS REPORT")
    report.append("=" * 60)
    
    # Overall stats
    total_zones = len(zone_stats)
    max_zone = zone_stats["current_zone_idx"].max()
    avg_crash = zone_stats["crash_rate"].mean()
    
    report.append(f"\n📊 OVERALL STATISTICS:")
    report.append(f"   Total zones with data: {total_zones}")
    report.append(f"   Max zone reached: {max_zone}")
    report.append(f"   Average crash rate: {avg_crash:.1f}%")
    
    # Acceleration zone info
    report.append(f"\n🚀 ACCELERATION ZONES (progressive threshold):")
    report.append(f"   Zone 0-30:   No crash detection (launch)")
    report.append(f"   Zone 31-100: Crash threshold = 3 km/h")
    report.append(f"   Zone 101-200: Crash threshold = 5 km/h")
    report.append(f"   Zone 201-400: Crash threshold = 7 km/h")
    report.append(f"   Zone 401-600: Crash threshold = 9 km/h")
    report.append(f"   Zone 601+:    Crash threshold = 10 km/h")
    
    # Kill zones summary (excluding acceleration zones)
    kill_zones = zone_stats[
        (zone_stats["crash_rate"] > 30) & 
        (zone_stats["current_zone_idx"] > 50)  # Only count real racing zones
    ].sort_values("crash_rate", ascending=False)
    
    report.append(f"\n🔴 KILL ZONES (>30% crash rate, zone 51+): {len(kill_zones)}")
    for _, row in kill_zones.head(5).iterrows():
        report.append(f"   Zone {int(row['current_zone_idx'])}: {row['crash_rate']:.1f}% crash rate")
    
    # Jump zones
    jump_zones = zone_stats[
        (zone_stats["airborne_rate"] > 50) & 
        (zone_stats["crash_rate"] < 20)
    ]
    if len(jump_zones) > 0:
        report.append(f"\n✈️ JUMP ZONES (high airborne, low crash): {len(jump_zones)}")
        for _, row in jump_zones.head(3).iterrows():
            report.append(f"   Zone {int(row['current_zone_idx'])}: {row['airborne_rate']:.1f}% airborne, {row['crash_rate']:.1f}% crash")
    
    # Recommendations
    report.append(f"\n💡 RECOMMENDATIONS:")
    if len(kill_zones) > 0:
        first_kill = int(kill_zones.iloc[0]["current_zone_idx"])
        spawn_zone = max(0, first_kill - 10)
        report.append(f"   → Set FORCE_SPAWN_ZONE={spawn_zone} to practice Zone {first_kill}")
        report.append(f"   → Ensure TRAINING_MODE='focused' is set")
    else:
        report.append(f"   → No critical kill zones! Consider advancing frontier.")
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.write(report_text)


def main():
    """Main entry point for section analysis."""
    output_dir = "analysis_results_section"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Extract features
    df = extract_features(df)
    
    # Run analyses
    zone_stats = analyze_zone_performance(df, output_dir)
    if zone_stats is None:
        return
    
    analyze_zone_transitions(df, output_dir)
    generate_report(zone_stats, output_dir)
    
    print(f"\n✅ Analysis complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()