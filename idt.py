import json
import pandas as pd
import numpy as np
import os

# ==== PARAMETERS ====
DISPERSION_THRESHOLD = 80     # pixels — how tightly points must cluster to count as one fixation
MIN_FIXATION_DURATION = 0.1   # seconds (100 ms)
SAMPLING_RATE = 33            # samples per second from eye tracker


# ==== FUNCTION: DETECT FIXATIONS ====
def detect_fixations(gaze_points, disp_threshold=DISPERSION_THRESHOLD, 
                     min_duration=MIN_FIXATION_DURATION, sampling_rate=SAMPLING_RATE):
    """
    Detects fixations using the I-DT (Dispersion Threshold) algorithm.
    """
    rows = []
    invalid_points = 0

    # Filter out points with missing or invalid coordinates
    for g in gaze_points:
        coords = g.get("coords")
        if not coords or coords.get("x") is None or coords.get("y") is None:
            invalid_points += 1
            continue  # skip invalid entries
        
        rows.append({
            "timestamp": g.get("timestamp"),
            "x": coords.get("x"),
            "y": coords.get("y"),
            "word": g.get("word"),
            "channel": g.get("channel"),
            "speech_meta": g.get("speech_meta")
        })

    if not rows:
        print("⚠️ No valid gaze points found in this session.")
        return []

    print(f"ℹ️ Valid gaze points: {len(rows)} (skipped {invalid_points} invalid points)")

    df = pd.DataFrame(rows)
    
    fixations = []
    window_size = int(min_duration * sampling_rate)
    i = 0
    n = len(df)
    
    # Sliding window approach (I-DT)
    while i + window_size <= n:
        window = df.iloc[i:i + window_size]
        dispersion = (window[['x', 'y']].max() - window[['x', 'y']].min()).sum()

        if dispersion <= disp_threshold:
            j = i + window_size
            while j < n:
                new_window = df.iloc[i:j]
                new_dispersion = (new_window[['x', 'y']].max() - new_window[['x', 'y']].min()).sum()
                if new_dispersion > disp_threshold:
                    break
                j += 1

            fixation_window = df.iloc[i:j]
            fix_start = fixation_window["timestamp"].iloc[0]
            fix_end = fixation_window["timestamp"].iloc[-1]
            fix_dur = fix_end - fix_start

            if fix_dur >= min_duration:
                fix_x = fixation_window["x"].mean()
                fix_y = fixation_window["y"].mean()
                fix_n_points = len(fixation_window)
                fix_channels = list(pd.Series(fixation_window["channel"].dropna()).unique())

                # Combine words
                words = [w for w in fixation_window["word"] if w]
                fix_word = " ".join(sorted(set(words))) if words else None

                # Merge speech meta
                metas = [m for m in fixation_window["speech_meta"] if m]
                if metas:
                    start_times = [m["start"] for m in metas if "start" in m]
                    end_times = [m["end"] for m in metas if "end" in m]
                    fix_speech_meta = {
                        "start": min(start_times),
                        "end": max(end_times),
                        "conf_mean": np.mean([m["conf"] for m in metas if "conf" in m])
                    }
                else:
                    fix_speech_meta = None

                fixations.append({
                    "start_time": fix_start,
                    "end_time": fix_end,
                    "duration": fix_dur,
                    "x": fix_x,
                    "y": fix_y,
                    "word": fix_word,
                    "channel": fix_channels,
                    "num_points": fix_n_points,
                    "speech_meta": fix_speech_meta
                })
            i = j
        else:
            i += 1

    return fixations


# ==== MAIN ====
if __name__ == "__main__":
    input_path = r"C:\Users\S.S.T\Documents\VsCode\Gaze Data Collection\64\recordings\session_20251026_153846.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    all_fixations = []

    # Process ALL sessions in the JSON
    for session_idx, session in enumerate(data.get("session", [])):
        print(f"\n=== Processing Session {session_idx} ===")
        gaze_points = session.get("gaze_data", [])
        if not gaze_points:
            print(f"⚠️ Session {session_idx} has no gaze data, skipping.")
            continue

        fixations = detect_fixations(gaze_points)
        all_fixations.append({
            "session_index": session_idx,
            "session_id": session.get("session_id", f"session_{session_idx}"),
            "num_fixations": len(fixations),
            "fixations": fixations
        })
        print(f"✅ Session {session_idx}: {len(fixations)} fixations detected")

    # Save all sessions' fixations
    output_path = "output_fixations_all_sessions_new.json"
    with open(output_path, "w") as f:
        json.dump({"sessions": all_fixations}, f, indent=4)

    print(f"\n🎯 Done! Fixations for {len(all_fixations)} sessions saved to {output_path}")
