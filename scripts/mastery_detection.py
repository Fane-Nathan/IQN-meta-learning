#!/usr/bin/env python3
"""
Mastery Detection System for Linesight Curriculum

The Core Problem:
- AI sets records for zones 100-1150 repeatedly
- No mechanism to detect "this zone is mastered, move on"
- Result: 46% of training spent on first 6% of track

Solution: Define "Mastery" mathematically and detect it automatically.

Mastery Criteria (a zone is "mastered" when):
1. SPEED: Average speed >= target speed for that zone type
2. CONSISTENCY: Success rate >= threshold (e.g., 80%)
3. IMPROVEMENT: No significant records set in last N runs
4. STABILITY: Low variance in completion times

When the "mastery frontier" (highest mastered zone) stops advancing,
trigger spawn zone advancement.
"""

import os
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
import json

# ========================= CONFIGURATION =========================

# Mastery thresholds
MASTERY_SUCCESS_RATE = 0.75          # 75% success rate to consider mastered
MASTERY_SPEED_PERCENTILE = 0.6       # Speed should be >= 60th percentile of zone's potential
MASTERY_MIN_ATTEMPTS = 20            # Need at least 20 attempts to evaluate
MASTERY_RECORD_STALENESS = 50        # If no records in 50 runs, zone is "stale"

# Advancement thresholds
ADVANCEMENT_TRIGGER_RUNS = 100       # Check advancement every N runs
STAGNATION_RUNS = 200                # If frontier unchanged for N runs, force advance
FRONTIER_BUFFER = 100                # Spawn this many zones before frontier

# Track configuration
TOTAL_ZONES = 8000
ZONE_SEGMENT_SIZE = 500              # Analyze in segments of 500 zones

# State files
MASTERY_STATE_FILE = "mastery_state.json"
CURRICULUM_CONFIG_FILE = "curriculum_config.txt"


# ========================= MASTERY METRICS =========================

class MasteryTracker:
    """
    Tracks mastery metrics across training runs.
    
    Key insight: "Mastery" isn't just about reaching a zone - 
    it's about reaching it CONSISTENTLY and FAST.
    """
    
    def __init__(self):
        self.state = self._load_state()
        
    def _load_state(self):
        """Load persistent mastery state."""
        default_state = {
            "total_runs": 0,
            "last_frontier": 0,
            "frontier_history": [],
            "last_record_run": {},      # zone -> last run that set a record
            "zone_attempts": {},         # zone -> total attempts
            "zone_successes": {},        # zone -> successful passes
            "zone_best_times": {},       # zone -> best time to reach
            "spawn_zone": 0,
            "last_advancement_run": 0,
            "created_at": datetime.now().isoformat()
        }
        
        if os.path.exists(MASTERY_STATE_FILE):
            try:
                with open(MASTERY_STATE_FILE, "r") as f:
                    loaded = json.load(f)
                    default_state.update(loaded)
            except Exception as e:
                print(f"⚠️ Error loading mastery state: {e}")
        
        return default_state
    
    def _save_state(self):
        """Save mastery state."""
        with open(MASTERY_STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def update_from_rollout(self, rollout_data):
        """
        Update mastery metrics from a single rollout.
        
        Args:
            rollout_data: dict with keys:
                - max_zone: highest zone reached
                - zone_times: dict of zone -> time to reach
                - crashed: bool
                - records_set: list of zones where records were set
        """
        self.state["total_runs"] += 1
        run_id = self.state["total_runs"]
        
        max_zone = rollout_data.get("max_zone", 0)
        zone_times = rollout_data.get("zone_times", {})
        records_set = rollout_data.get("records_set", [])
        
        # Update zone attempts and successes
        for zone in range(max_zone + 1):
            zone_str = str(zone)
            self.state["zone_attempts"][zone_str] = \
                self.state["zone_attempts"].get(zone_str, 0) + 1
            self.state["zone_successes"][zone_str] = \
                self.state["zone_successes"].get(zone_str, 0) + 1
        
        # Update best times
        for zone, time_ms in zone_times.items():
            zone_str = str(zone)
            current_best = self.state["zone_best_times"].get(zone_str, float('inf'))
            if time_ms < current_best:
                self.state["zone_best_times"][zone_str] = time_ms
        
        # Track record runs
        for zone in records_set:
            self.state["last_record_run"][str(zone)] = run_id
        
        self._save_state()
        
        # Check if we should evaluate advancement
        if run_id % ADVANCEMENT_TRIGGER_RUNS == 0:
            return self.evaluate_advancement()
        
        return None
    
    def calculate_zone_mastery(self, zone, zone_stats_df=None):
        """
        Calculate mastery score for a single zone.
        
        Returns:
            mastery_score: 0.0 to 1.0 (1.0 = fully mastered)
            details: dict with breakdown
        """
        zone_str = str(zone)
        
        attempts = self.state["zone_attempts"].get(zone_str, 0)
        successes = self.state["zone_successes"].get(zone_str, 0)
        last_record = self.state["last_record_run"].get(zone_str, 0)
        current_run = self.state["total_runs"]
        
        # Insufficient data
        if attempts < MASTERY_MIN_ATTEMPTS:
            return 0.0, {"reason": "insufficient_attempts", "attempts": attempts}
        
        # Calculate metrics
        success_rate = successes / attempts if attempts > 0 else 0
        runs_since_record = current_run - last_record
        record_staleness = min(1.0, runs_since_record / MASTERY_RECORD_STALENESS)
        
        # Speed metric (if zone_stats available)
        speed_score = 0.5  # default
        if zone_stats_df is not None:
            zone_data = zone_stats_df[zone_stats_df["current_zone_idx"] == zone]
            if not zone_data.empty:
                avg_speed = zone_data["avg_speed"].values[0]
                # Compare to nearby zones to get relative performance
                nearby = zone_stats_df[
                    (zone_stats_df["current_zone_idx"] >= zone - 50) &
                    (zone_stats_df["current_zone_idx"] <= zone + 50)
                ]
                if not nearby.empty:
                    percentile = (nearby["avg_speed"] <= avg_speed).mean()
                    speed_score = percentile
        
        # Combined mastery score
        # Weight: 40% success rate, 30% speed, 30% staleness (no new records = mastered)
        mastery_score = (
            0.4 * min(1.0, success_rate / MASTERY_SUCCESS_RATE) +
            0.3 * speed_score +
            0.3 * record_staleness
        )
        
        details = {
            "attempts": attempts,
            "success_rate": success_rate,
            "runs_since_record": runs_since_record,
            "speed_score": speed_score,
            "record_staleness": record_staleness,
            "mastery_score": mastery_score,
            "is_mastered": mastery_score >= 0.8
        }
        
        return mastery_score, details
    
    def find_mastery_frontier(self, zone_stats_df=None):
        """
        Find the highest zone that is "mastered".
        
        The mastery frontier is the highest zone where:
        - All zones before it have mastery_score >= 0.8
        - OR there's a consistent 80%+ success rate up to that point
        
        Returns:
            frontier_zone: highest mastered zone
            analysis: dict with details
        """
        max_attempted = max(
            [int(z) for z in self.state["zone_attempts"].keys()] or [0]
        )
        
        # Scan zones to find frontier
        frontier = 0
        consecutive_unmastered = 0
        zone_masteries = []
        
        for zone in range(0, min(max_attempted + 1, TOTAL_ZONES)):
            mastery, details = self.calculate_zone_mastery(zone, zone_stats_df)
            zone_masteries.append((zone, mastery, details))
            
            if mastery >= 0.7:  # Reasonably mastered
                frontier = zone
                consecutive_unmastered = 0
            else:
                consecutive_unmastered += 1
                
            # If we hit 50 consecutive unmastered zones, stop
            if consecutive_unmastered >= 50:
                break
        
        analysis = {
            "frontier": frontier,
            "max_attempted": max_attempted,
            "zones_analyzed": len(zone_masteries),
            "average_mastery": np.mean([m for _, m, _ in zone_masteries]) if zone_masteries else 0
        }
        
        return frontier, analysis
    
    def evaluate_advancement(self, zone_stats_df=None):
        """
        Determine if spawn zone should be advanced.
        
        Decision logic:
        1. Find current mastery frontier
        2. If frontier > current spawn + buffer, advance spawn
        3. If frontier stagnant for N runs, force advance
        
        Returns:
            dict with advancement decision
        """
        frontier, analysis = self.find_mastery_frontier(zone_stats_df)
        current_spawn = self.state["spawn_zone"]
        current_run = self.state["total_runs"]
        last_advancement = self.state["last_advancement_run"]
        
        # Track frontier history
        self.state["frontier_history"].append({
            "run": current_run,
            "frontier": frontier
        })
        # Keep last 100 entries
        self.state["frontier_history"] = self.state["frontier_history"][-100:]
        
        decision = {
            "should_advance": False,
            "current_spawn": current_spawn,
            "current_frontier": frontier,
            "recommended_spawn": current_spawn,
            "reason": "no_change_needed"
        }
        
        # Check if frontier has significantly advanced beyond spawn
        if frontier > current_spawn + FRONTIER_BUFFER + 200:
            new_spawn = frontier - FRONTIER_BUFFER
            decision.update({
                "should_advance": True,
                "recommended_spawn": new_spawn,
                "reason": f"Frontier ({frontier}) significantly ahead of spawn ({current_spawn})"
            })
        
        # Check for stagnation
        elif current_run - last_advancement > STAGNATION_RUNS:
            # Check if frontier has moved in recent history
            recent_frontiers = [
                h["frontier"] for h in self.state["frontier_history"][-20:]
            ]
            if recent_frontiers:
                frontier_variance = np.std(recent_frontiers)
                frontier_progress = max(recent_frontiers) - min(recent_frontiers)
                
                if frontier_progress < 50:  # Less than 50 zones progress
                    # Force advance!
                    new_spawn = min(frontier + 200, TOTAL_ZONES - 500)
                    decision.update({
                        "should_advance": True,
                        "recommended_spawn": new_spawn,
                        "reason": f"STAGNATION: Frontier stuck at ~{frontier} for {STAGNATION_RUNS} runs"
                    })
        
        # Apply advancement if decided
        if decision["should_advance"]:
            self.state["spawn_zone"] = decision["recommended_spawn"]
            self.state["last_frontier"] = frontier
            self.state["last_advancement_run"] = current_run
            self._write_curriculum_config(decision["recommended_spawn"])
        
        self._save_state()
        return decision
    
    def _write_curriculum_config(self, spawn_zone):
        """Write curriculum config file."""
        with open(CURRICULUM_CONFIG_FILE, "w") as f:
            f.write(f"FORCE_SPAWN_ZONE={int(spawn_zone)}\n")
            f.write("FOCUS_ZONE=None\n")
        print(f"📝 Updated {CURRICULUM_CONFIG_FILE}: FORCE_SPAWN_ZONE={spawn_zone}")
    
    def get_status_report(self, zone_stats_df=None):
        """Generate a human-readable status report."""
        frontier, analysis = self.find_mastery_frontier(zone_stats_df)
        
        report = []
        report.append("\n" + "="*60)
        report.append("🎯 MASTERY STATUS REPORT")
        report.append("="*60)
        report.append(f"Total runs:           {self.state['total_runs']}")
        report.append(f"Current spawn zone:   {self.state['spawn_zone']}")
        report.append(f"Mastery frontier:     Zone {frontier}")
        report.append(f"Track coverage:       {frontier / TOTAL_ZONES * 100:.1f}%")
        report.append(f"Zones analyzed:       {analysis['zones_analyzed']}")
        report.append(f"Average mastery:      {analysis['average_mastery']:.2f}")
        
        # Sample zone masteries
        report.append("\n📊 Sample Zone Mastery Scores:")
        sample_zones = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]
        for zone in sample_zones:
            if zone <= frontier + 500:
                score, details = self.calculate_zone_mastery(zone, zone_stats_df)
                status = "✅" if details.get("is_mastered") else "⏳"
                report.append(f"  Zone {zone:5d}: {score:.2f} {status} (SR: {details.get('success_rate', 0):.0%})")
        
        report.append("="*60 + "\n")
        return "\n".join(report)


# ========================= INTEGRATION HOOKS =========================

def parse_training_output(log_line):
    """
    Parse a training log line to extract rollout data.
    
    Expected formats:
    - "🚀 New Record for Zone 263: 3240ms (was 3280.0ms)"
    - "🏁 Start Zone: 19"
    - "Race time ratio   1.423519497106249"
    """
    rollout_data = {}
    
    if "New Record for Zone" in log_line:
        # Extract zone number
        import re
        match = re.search(r"Zone (\d+):", log_line)
        if match:
            zone = int(match.group(1))
            rollout_data["records_set"] = rollout_data.get("records_set", []) + [zone]
            rollout_data["max_zone"] = max(rollout_data.get("max_zone", 0), zone)
    
    return rollout_data


def integrate_with_collector(collector_instance):
    """
    Integration point for collector_process.py
    
    Add this to the collector's rollout completion callback:
    
    ```python
    from mastery_detection import MasteryTracker
    
    mastery_tracker = MasteryTracker()
    
    # After each rollout:
    rollout_data = {
        "max_zone": highest_zone_reached,
        "zone_times": zone_completion_times,
        "records_set": new_records_list
    }
    advancement = mastery_tracker.update_from_rollout(rollout_data)
    if advancement and advancement["should_advance"]:
        print(f"🎯 CURRICULUM ADVANCEMENT: {advancement['reason']}")
    ```
    """
    pass


# ========================= CLI =========================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mastery Detection System")
    parser.add_argument("--status", action="store_true", help="Show mastery status")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate advancement")
    parser.add_argument("--reset", action="store_true", help="Reset mastery state")
    parser.add_argument("--simulate", type=int, help="Simulate N rollouts for testing")
    parser.add_argument("--zone-stats", type=str, default="zone_stats.csv", 
                        help="Path to zone_stats.csv")
    
    args = parser.parse_args()
    
    tracker = MasteryTracker()
    
    # Load zone stats if available
    zone_stats_df = None
    if os.path.exists(args.zone_stats):
        zone_stats_df = pd.read_csv(args.zone_stats)
        print(f"📊 Loaded {len(zone_stats_df)} zones from {args.zone_stats}")
    
    if args.reset:
        if os.path.exists(MASTERY_STATE_FILE):
            os.remove(MASTERY_STATE_FILE)
        print("🔄 Mastery state reset")
        return
    
    if args.simulate:
        print(f"🎮 Simulating {args.simulate} rollouts...")
        import random
        for i in range(args.simulate):
            # Simulate rollout reaching random zone
            max_zone = random.randint(100, 1500)
            records = random.sample(range(100, max_zone), min(5, max_zone - 100))
            rollout = {
                "max_zone": max_zone,
                "records_set": records
            }
            result = tracker.update_from_rollout(rollout)
            if result and result["should_advance"]:
                print(f"\n🎯 Run {i+1}: {result['reason']}")
        print("\nSimulation complete!")
    
    if args.status or args.evaluate:
        print(tracker.get_status_report(zone_stats_df))
    
    if args.evaluate:
        decision = tracker.evaluate_advancement(zone_stats_df)
        print("\n🎯 ADVANCEMENT DECISION:")
        print(f"  Should advance: {decision['should_advance']}")
        print(f"  Current spawn:  {decision['current_spawn']}")
        print(f"  Recommended:    {decision['recommended_spawn']}")
        print(f"  Reason:         {decision['reason']}")


if __name__ == "__main__":
    main()
