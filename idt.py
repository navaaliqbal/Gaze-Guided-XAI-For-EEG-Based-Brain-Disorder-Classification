import json
import pandas as pd
import numpy as np
from collections import Counter
import os

# DISPERSION_THRESHOLD = 80
# MIN_FIXATION_DURATION = 0.1
# SAMPLING_RATE = 33
# for normal eeg
DISPERSION_THRESHOLD = 50
MIN_FIXATION_DURATION = 0.1
SAMPLING_RATE = 33


def detect_fixations(gaze_points, disp_threshold=DISPERSION_THRESHOLD,
                     min_duration=MIN_FIXATION_DURATION, sampling_rate=SAMPLING_RATE):

    # ==== FIX 1: safe extraction of coords ====
    df = pd.DataFrame([{
        "timestamp": g.get("timestamp"),
        "x": g.get("coords", {}).get("x") if g.get("coords") else None,
        "y": g.get("coords", {}).get("y") if g.get("coords") else None,
        "word": g.get("word"),
        "channel": g.get("channel"),
        "speech_meta": g.get("speech_meta")
    } for g in gaze_points if g is not None])

    fixations = []
    window_size = int(min_duration * sampling_rate)

    i = 0
    n = len(df)

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

                words = [w for w in fixation_window["word"] if w]
                fix_word = " ".join(sorted(set(words))) if words else None

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
    input_path = r"D:\FEB_fyprun\actual data\Khizar\normal\0000019\0000019.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    session = data["session"][0]
    gaze_points = session["gaze_data"]

    fixations = detect_fixations(gaze_points)

    # derive clean output file name
    base = os.path.basename(input_path)                      # session_20250930_094223.json
    stem = os.path.splitext(base)[0]                        # session_20250930_094223
    out_name = f"fixations_{stem}.json"                     # fixations_session_20250930_094223.json

    # output_path = os.path.join(
    #     os.path.dirname(input_path),                        # same folder as input
    #     out_name
    # )
    output_path= r"D:\FEB_fyprun\fixations_output\00000019_P0.json"

    # save JSON
    with open(output_path, "w") as f:
        json.dump({"fixations": fixations}, f, indent=4)

    print(f"Saved {len(fixations)} fixations → {output_path}")
