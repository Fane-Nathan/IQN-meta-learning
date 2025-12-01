#!/usr/bin/env python3
"""
Enhanced Curriculum Manager v2: Frontier-Based Progression

Problem with v1:
- Only detects "Kill Zones" (high crash rate zones)
- Doesn't detect "Frontier Zones" (zones AI never reaches)
- Can't advance beyond the exploration boundary

Solution:
- Track the "frontier" = highest zone AI consistently reaches
- If frontier hasn't advanced in N iterations, force spawn beyond it
- Use progressive spawning: spawn at (frontier - buffer) to push forward
"""

import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from config_files import config
except ImportError:
    # Fallback defaults
    class config:
        TRAINING_MODE = "focused"
        map_path = "ESL-Hockolicious.Challenge.Gbx"

# ========================= CONFIGURATION =========================
CURRICULUM_CONFIG_PATH = "curriculum_config.txt"
FRONTIER_STATE_PATH = "curriculum_frontier_state.txt"

# Frontier detection parameters
MIN_VISITS_FOR_FRONTIER = 10       # Zone needs at least N visits to count
FRONTIER_PERCENTILE = 95            # Use 95th percentile of reached zones
STAGNATION_THRESHOLD = 3            # Force advance after N stagnant iterations
SPAWN_BUFFER_ZONES = 50             # Spawn this many zones BEFORE frontier
ADVANCEMENT_STEP = 100              # When forcing advance, jump this many zones
MIN_SPAWN_ZONE = 20                 # Minimum allowable spawn zone (Map Start)

# Track parameters (should match your map)
TOTAL_ZONES = 8000                  # Total zones in your track
MAX_SPAWN_ZONE = 7500               # Don't spawn past this


# ========================= STATE TRACKING =========================

def load_frontier_state():
    """Load persistent frontier tracking state."""
    state = {
        "last_frontier": 0,
        "stagnation_count": 0,
        "forced_spawn_zone": None,
        "last_update": None,
        "history": []
    }
    
    if os.path.exists(FRONTIER_STATE_PATH):
        try:
            with open(FRONTIER_STATE_PATH, "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        if key in state:
                            if key in ["last_frontier", "stagnation_count", "forced_spawn_zone"]:
                                state[key] = int(value) if value != "None" else None
                            elif key == "history":
                                state[key] = [int(x) for x in value.split(",") if x]
                            else:
                                state[key] = value
        except Exception as e:
            print(f"⚠️ Error loading frontier state: {e}")
    
    return state


def save_frontier_state(state):
    """Save frontier tracking state."""
    with open(FRONTIER_STATE_PATH, "w") as f:
        f.write(f"last_frontier={state['last_frontier']}\n")
        f.write(f"stagnation_count={state['stagnation_count']}\n")
        f.write(f"forced_spawn_zone={state['forced_spawn_zone']}\n")
        f.write(f"last_update={datetime.now().isoformat()}\n")
        f.write(f"history={','.join(map(str, state['history'][-20:]))}\n")


# ========================= ANALYSIS FUNCTIONS =========================

def load_zone_data(zone_stats_path="zone_stats.csv"):
    """Load zone statistics from CSV."""
    if not os.path.exists(zone_stats_path):
        # Try alternative paths
        alt_paths = [
            "analysis_results_section/zone_stats.csv", # New path from overhauled script
            "analysis_results_curriculum/zone_stats.csv",
            "scripts/zone_stats.csv",
            "../zone_stats.csv"
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                zone_stats_path = alt
                break
        else:
            print("❌ No zone_stats.csv found!")
            return None
    
    try:
        df = pd.read_csv(zone_stats_path)
        print(f"📊 Loaded {len(df)} zones from {zone_stats_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading zone stats: {e}")
        return None


def detect_frontier(df, min_visits=MIN_VISITS_FOR_FRONTIER, percentile=FRONTIER_PERCENTILE):
    """
    Detect the exploration frontier - the highest zone AI consistently reaches.
    
    Args:
        df: DataFrame with zone statistics
        min_visits: Minimum visits for a zone to count
        percentile: Use this percentile of visited zones as frontier
    
    Returns:
        frontier_zone: The detected frontier zone
        stats: Dictionary with analysis details
    """
    # Filter to zones with enough visits
    visited = df[df["total_frames"] >= min_visits].copy()
    
    if visited.empty:
        return 0, {"error": "No zones with sufficient visits"}
    
    # Get zone indices
    zones = visited["current_zone_idx"].values
    
    # Calculate frontier metrics
    max_zone = int(zones.max())
    p95_zone = int(np.percentile(zones, percentile))
    median_zone = int(np.median(zones))
    
    # Weighted frontier: higher zones with more visits matter more
    weights = visited["total_frames"].values
    weighted_avg = int(np.average(zones, weights=weights))
    
    # Use the 95th percentile as the frontier (conservative)
    frontier = p95_zone
    
    stats = {
        "max_zone": max_zone,
        "p95_zone": p95_zone,
        "median_zone": median_zone,
        "weighted_avg": weighted_avg,
        "total_visited_zones": len(visited),
        "frontier": frontier
    }
    
    return frontier, stats


def detect_kill_zones(df, crash_threshold=20, min_visits=300):
    """
    Find zones with high crash rates (original curriculum logic).
    
    Returns:
        List of (zone_id, crash_rate) tuples sorted by zone
    """
    candidates = df[
        (df["crash_rate"] > crash_threshold) & 
        (df["total_frames"] > min_visits) & 
        (df["current_zone_idx"] > 20)  # Skip spawn area
    ]
    
    if candidates.empty:
        return []
    
    # Sort by zone (find earliest problem)
    sorted_zones = candidates.sort_values("current_zone_idx")
    
    return [
        (int(row["current_zone_idx"]), float(row["crash_rate"]))
        for _, row in sorted_zones.head(5).iterrows()
    ]


# ========================= CURRICULUM LOGIC =========================

def calculate_spawn_zone(frontier, state, kill_zones):
    """
    Determine the optimal spawn zone based on:
    1. Current frontier
    2. Stagnation count
    3. Kill zones
    
    Returns:
        spawn_zone: Zone to spawn at (or None for default)
        reason: Explanation string
    """
    # Check if we have any kill zones before the frontier
    early_kill_zone = None
    for zone_id, crash_rate in kill_zones:
        if zone_id < frontier:
            early_kill_zone = (zone_id, crash_rate)
            break
    
    # If there's a kill zone, focus on it first
    if early_kill_zone:
        spawn = max(MIN_SPAWN_ZONE, early_kill_zone[0] - 10)
        return spawn, f"Kill Zone detected at {early_kill_zone[0]} ({early_kill_zone[1]:.1f}% crash rate)"
    
    # Check for stagnation
    prev_frontier = state["last_frontier"]
    frontier_improved = frontier > prev_frontier + 20  # Meaningful improvement
    
    if frontier_improved:
        # Good progress! Reset stagnation, spawn near new frontier
        state["stagnation_count"] = 0
        spawn = max(MIN_SPAWN_ZONE, frontier - SPAWN_BUFFER_ZONES)
        return spawn, f"Frontier advanced to {frontier} (was {prev_frontier})"
    
    # Frontier stagnated
    state["stagnation_count"] += 1
    
    if state["stagnation_count"] >= STAGNATION_THRESHOLD:
        # Force advancement!
        forced_zone = min(frontier + ADVANCEMENT_STEP, MAX_SPAWN_ZONE)
        state["stagnation_count"] = 0
        return forced_zone, f"FORCED ADVANCE: Stagnant at {frontier} for {STAGNATION_THRESHOLD} iterations"
    
    # Normal case: spawn near frontier
    spawn = max(MIN_SPAWN_ZONE, frontier - SPAWN_BUFFER_ZONES)
    return spawn, f"Frontier at {frontier} (stagnation: {state['stagnation_count']}/{STAGNATION_THRESHOLD})"


def write_curriculum_config(spawn_zone, focus_zone=None, mode="focused"):
    """Write the curriculum configuration file."""
    with open(CURRICULUM_CONFIG_PATH, "w") as f:
        if spawn_zone is not None:
            f.write(f"FORCE_SPAWN_ZONE={int(spawn_zone)}\n")
        else:
            f.write("FORCE_SPAWN_ZONE=None\n")
        
        if focus_zone is not None:
            f.write(f"FOCUS_ZONE={int(focus_zone)}\n")
        else:
            f.write("FOCUS_ZONE=None\n")
    
    print(f"📝 Updated {CURRICULUM_CONFIG_PATH}")


# ========================= MAIN TEACHER LOGIC =========================

def run_teacher_step():
    """
    Main curriculum update function.
    
    Flow:
    1. Load zone statistics
    2. Detect current frontier
    3. Check for kill zones
    4. Determine spawn zone based on progress
    5. Update curriculum config
    """
    print("\n" + "="*60)
    print("🎓 CURRICULUM TEACHER v2: Frontier-Based Progression")
    print("="*60)
    
    # Load state
    state = load_frontier_state()
    print(f"\n📂 Previous frontier: Zone {state['last_frontier']}")
    print(f"📂 Stagnation count: {state['stagnation_count']}")
    
    # Run fresh analysis using the new script
    try:
        import section_analysis
        print("\n📊 Running Section Analysis...")
        section_analysis.main()
    except Exception as e:
        print(f"⚠️ Failed to run section_analysis: {e}")

    # Load zone data
    df = load_zone_data()
    if df is None:
        print("❌ Cannot proceed without zone data")
        return
    
    # Detect frontier
    frontier, stats = detect_frontier(df)
    print(f"\n🔍 FRONTIER ANALYSIS:")
    print(f"   Max zone reached: {stats.get('max_zone', 'N/A')}")
    print(f"   95th percentile:  {stats.get('p95_zone', 'N/A')}")
    print(f"   Median zone:      {stats.get('median_zone', 'N/A')}")
    print(f"   Current frontier: {frontier}")
    print(f"   Track coverage:   {frontier / TOTAL_ZONES * 100:.1f}%")
    
    # Detect kill zones
    kill_zones = detect_kill_zones(df)
    if kill_zones:
        print(f"\n⚠️ KILL ZONES DETECTED:")
        for zone_id, crash_rate in kill_zones[:3]:
            print(f"   Zone {zone_id}: {crash_rate:.1f}% crash rate")
    
    # Calculate spawn zone
    spawn_zone, reason = calculate_spawn_zone(frontier, state, kill_zones)
    
    print(f"\n🎯 CURRICULUM DECISION:")
    print(f"   {reason}")
    print(f"   → Spawn Zone: {spawn_zone}")
    
    # Update state
    state["last_frontier"] = frontier
    state["history"].append(frontier)
    state["forced_spawn_zone"] = spawn_zone
    save_frontier_state(state)
    
    # Write config
    training_mode = getattr(config, "TRAINING_MODE", "focused")
    
    if training_mode == "hybrid":
        # Hybrid mode: always start at 0, use FOCUS_ZONE for instant replay
        write_curriculum_config(None, focus_zone=spawn_zone, mode="hybrid")
        print(f"\n📋 Hybrid Mode: FOCUS_ZONE={spawn_zone} for Instant Replay")
    else:
        # Focused mode: spawn at the calculated zone
        write_curriculum_config(spawn_zone, focus_zone=None, mode="focused")
        print(f"\n📋 Focused Mode: FORCE_SPAWN_ZONE={spawn_zone}")
    
    print("\n" + "="*60)
    print("🎓 Teacher session complete!")
    print("="*60 + "\n")
    
    return {
        "frontier": frontier,
        "spawn_zone": spawn_zone,
        "reason": reason,
        "stats": stats
    }


# ========================= UTILITY FUNCTIONS =========================

def manual_set_spawn(zone):
    """Manually set the spawn zone (for debugging/testing)."""
    print(f"🔧 Manually setting spawn zone to {zone}")
    write_curriculum_config(zone)
    
    state = load_frontier_state()
    state["forced_spawn_zone"] = zone
    save_frontier_state(state)


def reset_curriculum():
    """Reset curriculum to initial state."""
    print("🔄 Resetting curriculum...")
    
    write_curriculum_config(None, None)
    
    if os.path.exists(FRONTIER_STATE_PATH):
        os.remove(FRONTIER_STATE_PATH)
    
    print("✅ Curriculum reset to default (Zone 0 start)")


def show_status():
    """Display current curriculum status."""
    print("\n📊 CURRICULUM STATUS")
    print("-" * 40)
    
    state = load_frontier_state()
    print(f"Last frontier:    Zone {state['last_frontier']}")
    print(f"Stagnation count: {state['stagnation_count']}")
    print(f"Forced spawn:     {state['forced_spawn_zone']}")
    print(f"History:          {state['history'][-5:]}")
    
    if os.path.exists(CURRICULUM_CONFIG_PATH):
        print(f"\n{CURRICULUM_CONFIG_PATH}:")
        with open(CURRICULUM_CONFIG_PATH, "r") as f:
            print(f.read())


# ========================= CLI =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curriculum Manager v2")
    parser.add_argument("--run", action="store_true", help="Run teacher step")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--reset", action="store_true", help="Reset curriculum")
    parser.add_argument("--set-spawn", type=int, help="Manually set spawn zone")
    parser.add_argument("--total-zones", type=int, default=8000, help="Total zones in track")
    
    args = parser.parse_args()
    
    if args.total_zones:
        TOTAL_ZONES = args.total_zones
    
    if args.status:
        show_status()
    elif args.reset:
        reset_curriculum()
    elif args.set_spawn is not None:
        manual_set_spawn(args.set_spawn)
    else:
        # Default: run teacher step
        run_teacher_step()
