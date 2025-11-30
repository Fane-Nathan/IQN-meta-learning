#!/usr/bin/env python3
"""
Mastery Monitor - Standalone Curriculum Advancement

This script monitors Linesight training output and automatically advances
the spawn zone when the AI has "mastered" the current section.

HOW IT WORKS:
1. Watches training output for "New Record" lines
2. Tracks which zones are getting records vs which are stable
3. Detects when the "mastery frontier" stops advancing
4. Updates curriculum_config.txt to force spawn advancement

USAGE:
    # Run in a separate terminal while training:
    python mastery_monitor.py --watch
    
    # Or pipe training output:
    python train.py 2>&1 | python mastery_monitor.py --pipe
    
    # Check current status:
    python mastery_monitor.py --status

NO MODIFICATION TO COLLECTOR REQUIRED!
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict, deque

# ========================= CONFIGURATION =========================

CURRICULUM_CONFIG_FILE = "curriculum_config.txt"
MASTERY_STATE_FILE = "mastery_monitor_state.json"

# Mastery detection parameters
RECORD_WINDOW = 100           # Track records over last N rollouts
MASTERY_THRESHOLD = 0.1       # Zone is mastered if <10% of rollouts set records
STAGNATION_ROLLOUTS = 150     # Force advance if no frontier movement in N rollouts
ADVANCEMENT_STEP = 300        # When advancing, jump this many zones ahead
FRONTIER_BUFFER = 50          # Spawn this far behind the frontier

# Track configuration  
TOTAL_ZONES = 8000


# ========================= STATE MANAGEMENT =========================

class MasteryMonitor:
    def __init__(self):
        self.state = self._load_state()
        self.recent_records = deque(maxlen=RECORD_WINDOW)
        
    def _load_state(self):
        default = {
            "total_rollouts": 0,
            "current_spawn": 0,
            "current_frontier": 0,
            "frontier_history": [],
            "zone_record_counts": {},
            "last_advancement_rollout": 0,
            "last_update": None
        }
        
        if os.path.exists(MASTERY_STATE_FILE):
            try:
                with open(MASTERY_STATE_FILE, "r") as f:
                    loaded = json.load(f)
                    default.update(loaded)
            except:
                pass
        
        return default
    
    def _save_state(self):
        self.state["last_update"] = datetime.now().isoformat()
        with open(MASTERY_STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def process_line(self, line):
        """Process a single line of training output."""
        
        # Detect new records
        record_match = re.search(r"New Record for Zone (\d+):", line)
        if record_match:
            zone = int(record_match.group(1))
            self._record_new_record(zone)
            return
        
        # Detect rollout completion (race time ratio line)
        if "Race time ratio" in line:
            self._complete_rollout()
    
    def _record_new_record(self, zone):
        """Track a new record being set."""
        zone_str = str(zone)
        self.state["zone_record_counts"][zone_str] = \
            self.state["zone_record_counts"].get(zone_str, 0) + 1
        self.recent_records.append(zone)
        
        # Update frontier if this is a new high
        if zone > self.state["current_frontier"]:
            self.state["current_frontier"] = zone
    
    def _complete_rollout(self):
        """Called at the end of each rollout."""
        self.state["total_rollouts"] += 1
        rollout_num = self.state["total_rollouts"]
        
        # Track frontier history
        self.state["frontier_history"].append({
            "rollout": rollout_num,
            "frontier": self.state["current_frontier"]
        })
        # Keep last 200 entries
        self.state["frontier_history"] = self.state["frontier_history"][-200:]
        
        # Check for advancement every 20 rollouts
        if rollout_num % 20 == 0:
            self._check_advancement()
        
        self._save_state()
    
    def _check_advancement(self):
        """Determine if spawn zone should be advanced."""
        rollout_num = self.state["total_rollouts"]
        current_spawn = self.state["current_spawn"]
        current_frontier = self.state["current_frontier"]
        last_advancement = self.state["last_advancement_rollout"]
        
        should_advance = False
        reason = ""
        new_spawn = current_spawn
        
        # Check 1: Frontier significantly ahead of spawn
        if current_frontier > current_spawn + ADVANCEMENT_STEP + 100:
            should_advance = True
            new_spawn = current_frontier - FRONTIER_BUFFER
            reason = f"Frontier ({current_frontier}) far ahead of spawn ({current_spawn})"
        
        # Check 2: Stagnation detection
        elif rollout_num - last_advancement > STAGNATION_ROLLOUTS:
            # Check if frontier has actually moved
            if len(self.state["frontier_history"]) >= 50:
                recent = [h["frontier"] for h in self.state["frontier_history"][-50:]]
                older = [h["frontier"] for h in self.state["frontier_history"][-100:-50]]
                
                recent_max = max(recent) if recent else 0
                older_max = max(older) if older else 0
                
                if recent_max - older_max < 50:  # Less than 50 zones progress
                    should_advance = True
                    new_spawn = min(current_frontier + ADVANCEMENT_STEP, TOTAL_ZONES - 500)
                    reason = f"STAGNATION: Frontier stuck at ~{current_frontier} for {STAGNATION_ROLLOUTS}+ rollouts"
        
        # Check 3: Record staleness in current spawn region
        else:
            spawn_region_records = sum(
                1 for z in self.recent_records 
                if current_spawn <= z <= current_spawn + 200
            )
            total_recent = len(self.recent_records)
            
            if total_recent > 50 and spawn_region_records / total_recent < MASTERY_THRESHOLD:
                should_advance = True
                new_spawn = current_frontier - FRONTIER_BUFFER
                reason = f"Spawn region mastered (only {spawn_region_records}/{total_recent} recent records in zones {current_spawn}-{current_spawn+200})"
        
        if should_advance:
            self._advance_spawn(new_spawn, reason)
    
    def _advance_spawn(self, new_spawn, reason):
        """Update spawn zone and write config."""
        print(f"\n{'='*60}")
        print(f"🎯 CURRICULUM ADVANCEMENT TRIGGERED")
        print(f"{'='*60}")
        print(f"Reason: {reason}")
        print(f"Old spawn: Zone {self.state['current_spawn']}")
        print(f"New spawn: Zone {new_spawn}")
        print(f"Current frontier: Zone {self.state['current_frontier']}")
        print(f"Track coverage: {self.state['current_frontier'] / TOTAL_ZONES * 100:.1f}%")
        print(f"{'='*60}\n")
        
        self.state["current_spawn"] = new_spawn
        self.state["last_advancement_rollout"] = self.state["total_rollouts"]
        
        # Write curriculum config
        with open(CURRICULUM_CONFIG_FILE, "w") as f:
            f.write(f"FORCE_SPAWN_ZONE={new_spawn}\n")
            f.write("FOCUS_ZONE=None\n")
        
        self._save_state()
    
    def get_status(self):
        """Generate status report."""
        lines = []
        lines.append("\n" + "="*60)
        lines.append("📊 MASTERY MONITOR STATUS")
        lines.append("="*60)
        lines.append(f"Total rollouts tracked:  {self.state['total_rollouts']}")
        lines.append(f"Current spawn zone:      {self.state['current_spawn']}")
        lines.append(f"Current frontier:        Zone {self.state['current_frontier']}")
        lines.append(f"Track coverage:          {self.state['current_frontier'] / TOTAL_ZONES * 100:.1f}%")
        lines.append(f"Last advancement:        Rollout #{self.state['last_advancement_rollout']}")
        
        # Recent record distribution
        if self.recent_records:
            records_by_segment = defaultdict(int)
            for z in self.recent_records:
                segment = (z // 500) * 500
                records_by_segment[segment] += 1
            
            lines.append(f"\n📈 Recent Records by Segment (last {len(self.recent_records)}):")
            for seg in sorted(records_by_segment.keys()):
                count = records_by_segment[seg]
                bar = "█" * (count // 2)
                lines.append(f"  {seg:5d}-{seg+499:5d}: {count:3d} {bar}")
        
        # Frontier history
        if self.state["frontier_history"]:
            recent = self.state["frontier_history"][-10:]
            lines.append(f"\n📉 Frontier History (last 10 checkpoints):")
            for h in recent:
                lines.append(f"  Rollout {h['rollout']:5d}: Zone {h['frontier']}")
        
        lines.append("="*60 + "\n")
        return "\n".join(lines)
    
    def force_advance(self, zone):
        """Manually force spawn zone advancement."""
        self._advance_spawn(zone, "Manual override")
    
    def reset(self):
        """Reset monitor state."""
        if os.path.exists(MASTERY_STATE_FILE):
            os.remove(MASTERY_STATE_FILE)
        if os.path.exists(CURRICULUM_CONFIG_FILE):
            os.remove(CURRICULUM_CONFIG_FILE)
        print("🔄 Monitor state reset")


# ========================= CLI =========================

def watch_mode(monitor):
    """Watch training log file for updates."""
    log_file = "training.log"  # Adjust path as needed
    
    print(f"👀 Watching for training output...")
    print(f"   (Reading from stdin - pipe your training output here)")
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        for line in sys.stdin:
            monitor.process_line(line)
            
            # Print status every 100 rollouts
            if monitor.state["total_rollouts"] % 100 == 0 and "Race time ratio" in line:
                print(f"📊 Rollout {monitor.state['total_rollouts']}: "
                      f"Frontier={monitor.state['current_frontier']}, "
                      f"Spawn={monitor.state['current_spawn']}")
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
        print(monitor.get_status())


def main():
    parser = argparse.ArgumentParser(description="Mastery Monitor for Linesight")
    parser.add_argument("--watch", action="store_true", 
                        help="Watch stdin for training output")
    parser.add_argument("--status", action="store_true",
                        help="Show current mastery status")
    parser.add_argument("--reset", action="store_true",
                        help="Reset monitor state")
    parser.add_argument("--advance", type=int,
                        help="Manually advance spawn to zone N")
    parser.add_argument("--total-zones", type=int, default=8000,
                        help="Total zones in track")
    
    args = parser.parse_args()
    
    global TOTAL_ZONES
    TOTAL_ZONES = args.total_zones
    
    monitor = MasteryMonitor()
    
    if args.reset:
        monitor.reset()
    elif args.advance:
        monitor.force_advance(args.advance)
    elif args.status:
        print(monitor.get_status())
    elif args.watch:
        watch_mode(monitor)
    else:
        # Default: show status
        print(monitor.get_status())
        print("Usage:")
        print("  python mastery_monitor.py --watch      # Monitor training")
        print("  python mastery_monitor.py --status     # Show status")
        print("  python mastery_monitor.py --advance N  # Force spawn to zone N")


if __name__ == "__main__":
    main()
