import time
import random
import os
MAX_GREEN_TIME = 60 #duration in seconds (cap per phase)
LANE_ORDER = [1,2,3,4]

# Average service time per vehicle (sec). Could be refined using empirical data.
AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE = 1

def _baseline_time_until_green(current_green_lane: int, time_left_sec: float):
    """Fixed schedule baseline: lanes served in natural order with 60s each.
    Returns per-lane time until next green begins.
    """
    order = [current_green_lane]
    for i in range(1,4):
        order.append(((current_green_lane - 1 + i) % 4) + 1)
    per_lane = [0.0,0.0,0.0,0.0]
    for pos, lane in enumerate(order):
        if pos == 0:
            per_lane[lane-1] = float(time_left_sec)
        else:
            per_lane[lane-1] = float(time_left_sec + MAX_GREEN_TIME*pos)
    return per_lane, order

def _adaptive_time_until_green(upcoming_order, time_left_sec: float, upcoming_durations):
    """Adaptive schedule: time until next green based on current remaining time and upcoming durations.
    upcoming_durations is aligned with upcoming_order and should have durations for lanes in seconds,
    with index 0 corresponding to the current lane's remaining time.
    Returns per-lane time until its green begins.
    """
    per_lane = [0.0,0.0,0.0,0.0]
    for pos, lane in enumerate(upcoming_order):
        if pos == 0:
            per_lane[lane-1] = float(time_left_sec)
        else:
            per_lane[lane-1] = float(time_left_sec + sum(upcoming_durations[1:pos]))
    return per_lane

# Legacy cumulative function removed; previous cumulative arrays deleted.
def _add_to_cumulative(*_args, **_kwargs):
    pass

def  traffic_control(Singleton, max_seconds=0, max_cycles=0):
    try:
        time.sleep(5)
        COUNTER = 0
        # Persistent pending queue per lane; counts only decrease when that lane gets green
        pending_queue=[0,0,0,0]
        # Track last total detections seen per lane to accumulate deltas while lanes are red
        last_seen_total=[0,0,0,0]
        green_lane_order = LANE_ORDER
        timeDurationForLane = 15
        start_ts = time.time()
        print()
        print(f"****************ADAPTIVE TRAFFIC CONTROL SYSTEM****************")
        while True:
            #started Green Light from Lane 1 at 9.00 A.M
            print()
            print(f"Green Light Is At Lane {green_lane_order[0]} currently")
            print()
            print(f"Green Lane Order :: {green_lane_order}")
            print()
        
            
            while(timeDurationForLane > 0):
                print(f"Time Left : {timeDurationForLane} seconds")
                # Update shared schedule for overlays (compute upcoming durations based on counts)
                upcoming_durations = [timeDurationForLane]  # remaining time for current lane
                for lane in green_lane_order[1:]:
                    cnt = pending_queue[lane-1]
                    dur = 0 if cnt <= 0 else min(AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE*cnt, MAX_GREEN_TIME)
                    upcoming_durations.append(dur)
                try:
                    Singleton.set_schedule(green_lane_order[0], timeDurationForLane, green_lane_order, upcoming_durations)
                except Exception:
                    pass
                if timeDurationForLane <= 5:
                    time.sleep(timeDurationForLane)
                    timeDurationForLane = 0
                    continue
                time.sleep(2)
                timeDurationForLane -= 5
            
            print()
            if(timeDurationForLane <= 0):
                print(f"Orange light at lane {green_lane_order[0]} for 3 second")
                current_lane = green_lane_order[0]
                # When a lane finishes green, its pending queue is served -> reset to 0
                pending_queue[current_lane-1] = 0
                # Update last seen total for the current lane to avoid recounting served vehicles
                try:
                    last_seen_total[current_lane-1] = Singleton.get_count(current_lane-1)
                except Exception:
                    pass
                print(f"Accumulating vehicles for lanes (red): {green_lane_order[1]}, {green_lane_order[2]}, {green_lane_order[3]}")
                # For other lanes, accumulate new arrivals since last checkpoint; do not reset
                for i in range(3):
                    lane = green_lane_order[i+1]
                    try:
                        total = Singleton.get_count(lane-1)
                    except Exception:
                        total = last_seen_total[lane-1]
                    delta = max(0, total - last_seen_total[lane-1])
                    pending_queue[lane-1] += delta
                    last_seen_total[lane-1] = total
                print()
                print(f"Pending vehicles per lane (queues): {pending_queue}")
            
            print()
            print(f"Green Light is Changed to Red Light for Lane {green_lane_order[0]}")
            print()
            #increasing Counter for counting the number of green lights in a sequence
            COUNTER+=1

            #changing green lane order based on vehicle count
            green_lane_order.append(green_lane_order.pop(0))
            while pending_queue[green_lane_order[0]-1] == 0:
                green_lane_order.append(green_lane_order.pop(0))
                COUNTER+=1
                if COUNTER>=4:
                    break
            #find lane with max vehicles from remaining red lights
            max_vehicle_lane=green_lane_order[0] 
            
            for i in range(1, 4-COUNTER):
                if(pending_queue[green_lane_order[i]-1] > pending_queue[max_vehicle_lane-1]):
                    max_vehicle_lane = green_lane_order[i]
            
            #swap lane with max vehicle with current first green order lane
            max_lane_ind = green_lane_order.index(max_vehicle_lane)
            green_lane_order[0],green_lane_order[max_lane_ind] = green_lane_order[max_lane_ind], green_lane_order[0]

            #calculate time duration for that lane
            timeDurationForLane = min(AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE*pending_queue[green_lane_order[0]-1], MAX_GREEN_TIME)
            if [0]*len(pending_queue) == pending_queue:
                timeDurationForLane = 15
            print(f" Green Light Time for lane {green_lane_order[0]} is {timeDurationForLane}sec")

            # Waiting time comparison (baseline fixed vs adaptive) as time until next green only
            base_t_until, base_order = _baseline_time_until_green(green_lane_order[0], timeDurationForLane)
            # For adaptive, reuse the last computed upcoming_durations but align index 0 to remaining time
            adaptive_durations = [timeDurationForLane]
            for lane in green_lane_order[1:]:
                cnt = pending_queue[lane-1]
                dur = 0 if cnt <= 0 else min(AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE*cnt, MAX_GREEN_TIME)
                adaptive_durations.append(dur)
            adapt_t_until = _adaptive_time_until_green(green_lane_order.copy(), timeDurationForLane, adaptive_durations)

            per_lane_saved = [round(max(0.0, base_t_until[i] - adapt_t_until[i]), 1) for i in range(4)]
            # Vehicle-weighted average saved seconds per vehicle this phase (bounded and non-cumulative)
            total_waiting = sum(pending_queue)
            weighted_avg_saved = 0.0
            if total_waiting > 0:
                weighted_avg_saved = round(sum(per_lane_saved[i] * pending_queue[i] for i in range(4)) / total_waiting, 1)

            print(f" Time until next green baseline (s): {[round(x,1) for x in base_t_until]}")
            print(f" Time until next green adaptive (s): {[round(x,1) for x in adapt_t_until]}")
            print(f" Per-lane saved time this phase (s): {per_lane_saved}")
            print(f" Weighted avg time saved per vehicle (this phase): {weighted_avg_saved}s")
            if COUNTER==4:
                green_lane_order=[1,2,3,4]
                timeDurationForLane=min(AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE*pending_queue[0], MAX_GREEN_TIME)

            # Graceful stop conditions
            elapsed = time.time() - start_ts
            if max_seconds > 0 and elapsed >= max_seconds:
                print(f"Reached max controller runtime {max_seconds}s. Shutting down controller.")
                break
            if max_cycles > 0 and COUNTER >= max_cycles:
                print(f"Reached max controller cycles {max_cycles}. Shutting down controller.")
                break
            if os.path.exists("STOP_ATCS"):
                print("STOP_ATCS flag file detected. Shutting down controller.")
                break
    except Exception as Error:
        print(Error)
    finally:
        print()
        print(f"****************ADAPTIVE TRAFFIC CONTROL SYSTEM****************")
        print("---------------Closed-----------------")
        return
