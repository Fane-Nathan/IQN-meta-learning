"""
This file implements a single multithreaded worker that handles a Trackmania game instance and provides rollout results to the learner process.
"""

import importlib
import time
from itertools import chain, count, cycle
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp

from config_files import config_copy
from trackmania_rl import utilities
from trackmania_rl.agents import iqn as iqn
from trackmania_rl.utilities import set_random_seed


def collector_process_fn(
    rollout_queue,
    uncompiled_shared_network,
    shared_network_lock,
    game_spawning_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tmi_port: int,
    process_number: int,
):
    print(f"DEBUG: Collector Process {process_number} starting...", flush=True)
    from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager
    from trackmania_rl.tmi_interaction import game_instance_manager
    
    # Add scripts to path for MasteryTracker
    import sys
    if str(base_dir / "scripts") not in sys.path:
        sys.path.append(str(base_dir / "scripts"))
    from mastery_detection import MasteryTracker

    set_random_seed(process_number)

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    mastery_tracker = MasteryTracker()

    inference_network, uncompiled_inference_network = iqn.make_untrained_iqn_network(config_copy.use_jit, is_inference=True)
    try:
        inference_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
    except Exception as e:
        print("Worker could not load weights, exception:", e)

    inferer = iqn.Inferer(inference_network, config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)

    def update_network():
        # Update weights of the inference network
        with shared_network_lock:
            uncompiled_inference_network.load_state_dict(uncompiled_shared_network.state_dict())

    # ========================================================
    # Training loop
    # ========================================================
    inference_network.train()

    map_cycle_str = str(config_copy.map_cycle)
    set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
    map_cycle_iter = cycle(chain(*config_copy.map_cycle))

    zone_centers_filename = None

    # ========================================================
    # Warmup pytorch and numba
    # ========================================================
    for _ in range(5):
        inferer.infer_network(
            np.random.randint(low=0, high=255, size=(1, config_copy.H_downsized, config_copy.W_downsized), dtype=np.uint8),
            np.random.rand(config_copy.float_input_dim).astype(np.float32),
        )
    # game_instance_manager.update_current_zone_idx(0, zone_centers, np.zeros(3))

    time_since_last_queue_push = time.perf_counter()
    for loop_number in count(1):
        importlib.reload(config_copy)

        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms
        tmi.running_speed = config_copy.running_speed

        # ===============================================
        #   DID THE CYCLE CHANGE ?
        # ===============================================

        if str(config_copy.map_cycle) != map_cycle_str:
            map_cycle_str = str(config_copy.map_cycle)
            set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
            map_cycle_iter = cycle(chain(*config_copy.map_cycle))

        # ===============================================
        #   GET NEXT MAP FROM CYCLE
        # ===============================================
        next_map_tuple = next(map_cycle_iter)
        if next_map_tuple[2] != zone_centers_filename:
            zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
        map_name, map_path, zone_centers_filename, is_explo, fill_buffer = next_map_tuple
        map_status = "trained" if map_name in set_maps_trained else "blind"

        inferer.epsilon = utilities.from_exponential_schedule(config_copy.epsilon_schedule, shared_steps.value)
        inferer.epsilon_boltzmann = utilities.from_exponential_schedule(config_copy.epsilon_boltzmann_schedule, shared_steps.value)
        inferer.tau_epsilon_boltzmann = config_copy.tau_epsilon_boltzmann
        inferer.is_explo = is_explo

        # ===============================================
        #   CURRICULUM LEARNING (The Student)
        # ===============================================
        start_state = None
        focus_zone = None
        start_zone_idx = None

        # TRAINING_MODE check MUST happen FIRST, BEFORE any file reading
        training_mode = getattr(config_copy, "TRAINING_MODE", "standard")
        print(f"DEBUG: Training Mode = {training_mode}")

        if training_mode == "hybrid":
            # HYBRID MODE: Always start at Zone 0, never load resurrection states
            # Only read FOCUS_ZONE for future Instant Replay feature
            print("DEBUG: Hybrid Mode - Enforcing Zone 0 start (no state resurrection)")
            try:
                config_path = base_dir / "curriculum_config.txt"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        for line in f:
                            if "FOCUS_ZONE" in line:
                                val = line.split("=")[1].strip()
                                if val != "None":
                                    focus_zone = int(float(val))
                                    print(f"DEBUG: Hybrid Mode - Focus Zone = {focus_zone} (for Instant Replay)")
            except Exception as e:
                print(f"Curriculum Read Error: {e}")
                
        elif training_mode == "focused":
            # FOCUSED MODE: Allow state resurrection from curriculum config
            print("DEBUG: Focused Mode - State resurrection enabled")
            try:
                config_path = base_dir / "curriculum_config.txt"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        for line in f:
                            if "FORCE_SPAWN_ZONE" in line:
                                val = line.split("=")[1].strip()
                                if val != "None":
                                    zone_idx = int(float(val))
                                    start_zone_idx = zone_idx
                                    state_path = base_dir / "states" / f"zone_{zone_idx}.pkl"
                                    if state_path.exists():
                                        import pickle
                                        with open(state_path, "rb") as sf:
                                            start_state = pickle.load(sf)
                                        print(f"DEBUG: Focused Mode - Loaded state for Zone {zone_idx}")
                                    else:
                                        print(f"⚠️ State file not found: {state_path}")
                            if "FOCUS_ZONE" in line:
                                val = line.split("=")[1].strip()
                                if val != "None":
                                    focus_zone = int(float(val))
            except Exception as e:
                print(f"Curriculum Read Error: {e}")
                
        else:
            # STANDARD MODE: No curriculum, always Zone 0
            print("DEBUG: Standard Mode - No curriculum active")

        if inference_network.training and not is_explo:
            inference_network.eval()
        elif is_explo and not inference_network.training:
            inference_network.train()

        update_network()

        rollout_start_time = time.perf_counter()
        
        # Pass start_state to rollout if supported (requires modifying GameInstanceManager too)
        # Since I can't easily modify GameInstanceManager signature without breaking things,
        # I will inject it via a side channel or modify GameInstanceManager to accept **kwargs?
        # Actually, I should modify GameInstanceManager.rollout signature.
        # But for now, let's assume I can't.
        # Wait, I CAN modify GameInstanceManager.
        # But let's check if I can just set it BEFORE calling rollout?
        # No, rollout resets the game.
        
        # Let's modify the call to pass start_state, and I will update GameInstanceManager next.
        rollout_results, end_race_stats = tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=map_path,
            zone_centers=zone_centers,
            update_network=update_network,
            epsilon=inferer.epsilon,
            epsilon_boltzmann=inferer.epsilon_boltzmann,
            start_state=start_state,
            save_states_dir=base_dir / "states",

            focus_zone=focus_zone, # Pass the focus zone for Instant Replay
            start_zone_idx=start_zone_idx,
        )
        rollout_end_time = time.perf_counter()
        rollout_duration = rollout_end_time - rollout_start_time
        rollout_results["worker_time_in_rollout_percentage"] = rollout_duration / (time.perf_counter() - time_since_last_queue_push)
        time_since_last_queue_push = time.perf_counter()
        
        # Mastery Detection Integration
        try:
            rollout_data_mastery = {
                "max_zone": rollout_results["furthest_zone_idx"],
                "zone_times": {}, # We don't have per-zone times easily accessible here without processing cp_times
                "records_set": [], # We'd need to parse logs or track this, for now let's rely on max_zone
                "crashed": not end_race_stats["race_finished"]
            }
            # If we have race time and finished, we can infer some speed metrics if needed
            
            advancement = mastery_tracker.update_from_rollout(rollout_data_mastery)
            if advancement and advancement["should_advance"]:
                print(f"🎯 CURRICULUM ADVANCEMENT: {advancement['reason']}")
        except Exception as e:
            print(f"Mastery Tracker Error: {e}")

        print("", flush=True)

        if not tmi.last_rollout_crashed:
            # Inject Teacher Stats for Data Mining
            try:
                import json
                mastery_state_path = base_dir / "scripts" / "mastery_monitor_state.json"
                if mastery_state_path.exists():
                    with open(mastery_state_path, "r") as f:
                        mastery_state = json.load(f)
                        teacher_frontier = mastery_state.get("current_frontier", 0)
                        teacher_spawn = mastery_state.get("current_spawn", 0)
                else:
                    teacher_frontier = 0
                    teacher_spawn = 0
            except Exception as e:
                print(f"Failed to read teacher stats: {e}")
                teacher_frontier = 0
                teacher_spawn = 0

            # Add to rollout results (replicated for each step to match dataframe length)
            n_steps = len(rollout_results["frames"])
            rollout_results["teacher_frontier"] = [teacher_frontier] * n_steps
            rollout_results["teacher_spawn_zone"] = [teacher_spawn] * n_steps

            rollout_queue.put(
                (
                    rollout_results,
                    end_race_stats,
                    True,  # fill buffer
                    False,  # is_explo
                    map_path,
                    "trained",
                    time.perf_counter() - rollout_start_time,
                    loop_number,
                )
            )

