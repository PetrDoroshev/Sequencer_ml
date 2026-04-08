import symusic
from symusic.core import ScoreTick, TrackTick
from symusic import Score
from pandas import DataFrame
from typing import List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np  

def get_bar_duration_ticks(score: ScoreTick) -> int:

    if not score.time_signatures:

        return score.ticks_per_quarter * 4

    ts = score.time_signatures[0]
 
    beats_per_bar = ts.numerator * 4 / ts.denominator
    return int(score.ticks_per_quarter * beats_per_bar)

def bar_duration_ticks(score: ScoreTick) -> int:

    ts = score.time_signatures[0] if score.time_signatures else None
    numerator = ts.numerator if ts else 4
    denominator = ts.denominator if ts else 4

    return int(score.ticks_per_quarter * 4 * numerator / denominator)

def quantize_bar(
    score: ScoreTick,
    track: TrackTick,
    bar_index: int,
    n_slots: int = 24
) -> Tuple[List[int], List[int]]:
  
    bar_ticks = get_bar_duration_ticks(score)
    bar_start = bar_index * bar_ticks
    bar_end = bar_start + bar_ticks

    onsets = [0] * n_slots
    pitches = [0] * n_slots

    for note in track.notes:
        if note.time < bar_start or note.time >= bar_end:
            continue
        pos = note.time - bar_start
        slot = int(pos * n_slots / bar_ticks)
  
        slot = max(0, min(slot, n_slots - 1))
        onsets[slot] = 1

        if note.pitch > pitches[slot]:
            pitches[slot] = note.pitch

    return onsets, pitches

def toussaint_weights(n_slots: int = 24) -> List[float]:

    weights = []
    for t in range(n_slots):
        if t % 6 == 0:
            weights.append(1.0)
        elif t % 3 == 0:
            weights.append(2.0)
        elif t % 2 == 0:
            weights.append(3.0)
        else:
            weights.append(4.0)
    return weights


def rhythmic_complexity_for_bar(
    onsets: List[int],
    weights: Optional[List[float]] = None
) -> float:

    n = len(onsets)
    if weights is None:
        weights = toussaint_weights(n)
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    weighted = sum(o * w for o, w in zip(onsets, weights))
    return weighted / total_weight

def pitch_range_for_bar(
    pitches: List[int],
    r_norm: float
) -> float:
    
    if not pitches:
        return 0.0
    return (max(pitches) - min(pitches)) / r_norm

def note_density_for_bar(
    score: ScoreTick,
    track: TrackTick, 
    min_note_fraction=1/16
) -> float:
    
    if track.note_num() == 0:
        return 0.0

    bar_ticks = bar_duration_ticks(score)
    duration_ticks = track.end()
    num_bars = duration_ticks / bar_ticks

    density = track.note_num() / num_bars

    max_notes_per_bar = 1.0 / min_note_fraction
    return min(density / max_notes_per_bar, 1.0)


def contour_for_bar(
    track: TrackTick,
    r_norm: float
) -> float:
  
    notes = sorted(track.notes, key=lambda n: n.time)
    if len(notes) < 2:
        return 0.0
    directions = [1 if notes[i+1].pitch > notes[i].pitch else -1 if notes[i+1].pitch < notes[i].pitch else 0 for i in range(len(notes)-1)]
    return sum(directions) / len(directions)

def compute_attributes_for_track(
    score: ScoreTick,
    track: TrackTick,
    r_norm: float,
    n_slots: int = 24,
    include_rhythmic: bool = False
) -> dict:
    
    bar_ticks = get_bar_duration_ticks(score)
    total_ticks = track.end()
    n_bars = (total_ticks + bar_ticks - 1) // bar_ticks

    sums = {
        "pitch_range": 0.0,
        "note_density": 0.0,
        "contour": 0.0,
    }

    for bar_idx in range(n_bars):
        onsets, pitches = quantize_bar(score, track, bar_idx, n_slots)
        sums["pitch_range"] += pitch_range_for_bar(pitches, r_norm)
        sums["note_density"] += note_density_for_bar(score, track)
        sums["contour"] += contour_for_bar(track, r_norm)
        #sums['rythmic_complexity'] += rhythmic_complexity_for_bar(onsets)
       
    if n_bars == 0:
        return {k: 0.0 for k in sums}
    return {k: v / n_bars for k, v in sums.items()}

def plot_attributes(csv_path: Path, attr_dict):

    df = pd.read_csv(csv_path)
    dict_df = df.to_dict()

    for key in dict_df.keys():

        attr_dict[key] = list(dict_df[key].values())
    
    plt.hist(attr_dict["pitch_range"], bins=10)
    plt.title("Распределение атрибута")
    plt.show()

    plt.hist(attr_dict["note_density"], bins=10)
    plt.title("Распределение атрибута")
    plt.show()

    '''
    plt.hist(attr_dict["rythmic_complexity"], bins=20)
    plt.title("Распределение атрибута")
    plt.show()
    '''

    plt.hist(attr_dict["contour"], bins=10)
    plt.title("Распределение атрибута")
    plt.show()


if __name__ == "__main__":

    from midi_pp import directory_min_max_pitch

    CSV_PATH = Path("./vgmusic_aug.csv")

    MIDIS_DIR = Path("./ds/vgmusic_midi/undersampled_aug")

    MIN_PITCH, MAX_PITCH = directory_min_max_pitch(MIDIS_DIR)
    PITCH_DIFF = MAX_PITCH - MIN_PITCH

    attr_dict = {"path": [], 
                 "pitch_range": [],
                 "note_density": [],
                 "contour": []}
    
    for file in tqdm(list(MIDIS_DIR.glob("**/*.mid"))):

        score = Score(file.resolve())
        attributes = compute_attributes_for_track(score, score.tracks[0], PITCH_DIFF,
                            n_slots=24)

        full_filename = str(Path(file))  

        attr_dict["path"].append(full_filename)

        for attribute in attributes.keys():

            attr_dict[attribute].append(attributes[attribute])
    
    
    df = DataFrame(attr_dict)
    df.to_csv(CSV_PATH, header=True, index=False)
    
    
    plot_attributes(CSV_PATH, attr_dict)

