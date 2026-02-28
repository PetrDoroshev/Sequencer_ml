from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from symusic import Score, Track


def _get_bar_duration_ticks(score: Score) -> int:
    if not score.time_signatures:
        numerator = 4
        denominator = 4
    else:
        ts = score.time_signatures[0]
        numerator = ts.numerator
        denominator = ts.denominator

    return score.ticks_per_quarter * numerator * 4 // denominator


def _track_bars_count(score: Score, track: Track) -> int:
    if track.note_num() == 0:
        return 0

    bar_ticks = _get_bar_duration_ticks(score)
    return (track.end() + bar_ticks - 1) // bar_ticks


def default_toussaint_weights(n: int = 24) -> List[float]:
    if n <= 0:
        return []
    weights = []
    for t in range(n):
        if t % 6 == 0:
            weights.append(1.0)
        elif t % 3 == 0:
            weights.append(2.0)
        elif t % 2 == 0:
            weights.append(3.0)
        else:
            weights.append(4.0)
    return weights


def _quantize_bar_onsets(
    score: Score,
    track: Track,
    bar_index: int,
    slots: int,
) -> Tuple[List[int], List[int]]:
    bar_ticks = _get_bar_duration_ticks(score)
    bar_start = bar_index * bar_ticks
    bar_end = bar_start + bar_ticks

    onsets = [0] * slots
    midi_vals = [0] * slots

    for note in track.notes:
        if note.time < bar_start or note.time >= bar_end:
            continue
        pos = note.time - bar_start
        slot = int(pos * slots / bar_ticks)
        if slot < 0:
            slot = 0
        elif slot >= slots:
            slot = slots - 1

        onsets[slot] = 1
        if note.pitch > midi_vals[slot]:
            midi_vals[slot] = int(note.pitch)

    return onsets, midi_vals


def rhythmic_complexity(
    score: Score,
    track: Track,
    bar_index: int,
    slots: int = 24,
    weights: Optional[Sequence[float]] = None,
) -> float:
    if slots <= 0:
        return 0.0

    if weights is None:
        weights = default_toussaint_weights(slots)
    if len(weights) != slots:
        raise ValueError("weights length must match slots")

    onsets, _ = _quantize_bar_onsets(score, track, bar_index, slots)

    num = 0.0
    den = 0.0
    for t in range(slots):
        w = float(weights[t])
        den += w
        if onsets[t] == 1:
            num += w
    return 0.0 if den == 0 else num / den


def pitch_range(
    score: Score,
    track: Track,
    bar_index: int,
    r_norm: float,
    slots: int = 24,
) -> float:
    if r_norm <= 0:
        raise ValueError("r_norm must be > 0")

    _, midi_vals = _quantize_bar_onsets(score, track, bar_index, slots)
    if not midi_vals:
        return 0.0

    return (max(midi_vals) - min(midi_vals)) / r_norm


def note_density(
    score: Score,
    track: Track,
    bar_index: int,
    slots: int = 24,
) -> float:
    if slots <= 0:
        return 0.0

    onsets, _ = _quantize_bar_onsets(score, track, bar_index, slots)
    return sum(onsets) / slots


def contour(
    score: Score,
    track: Track,
    bar_index: int,
    r_norm: float,
    slots: int = 24,
) -> float:
    if r_norm <= 0:
        raise ValueError("r_norm must be > 0")

    _, midi_vals = _quantize_bar_onsets(score, track, bar_index, slots)
    if len(midi_vals) < 2:
        return 0.0

    total = 0.0
    for t in range(len(midi_vals) - 1):
        total += (midi_vals[t + 1] - midi_vals[t])

    return total / r_norm


def compute_attributes_for_bar(
    score: Score,
    track: Track,
    bar_index: int,
    r_norm: float,
    slots: int = 24,
    weights: Optional[Sequence[float]] = None,
) -> dict:
    return {
        "rhythmic_complexity": rhythmic_complexity(score, track, bar_index, slots=slots, weights=weights),
        "pitch_range": pitch_range(score, track, bar_index, r_norm=r_norm, slots=slots),
        "note_density": note_density(score, track, bar_index, slots=slots),
        "contour": contour(score, track, bar_index, r_norm=r_norm, slots=slots),
    }


def compute_attributes_for_track(
    score: Score,
    track: Track,
    r_norm: float,
    slots: int = 24,
    weights: Optional[Sequence[float]] = None,
    aggregate: Optional[str] = None,
):
    bars = _track_bars_count(score, track)
    per_bar = [
        compute_attributes_for_bar(score, track, bar_index=i, r_norm=r_norm, slots=slots, weights=weights)
        for i in range(bars)
    ]

    if aggregate is None:
        return per_bar

    if aggregate not in {"mean", "weighted"}:
        raise ValueError("aggregate must be one of: None, 'mean', 'weighted'")

    if bars == 0:
        return {
            "rhythmic_complexity": 0.0,
            "pitch_range": 0.0,
            "note_density": 0.0,
            "contour": 0.0,
        }

    if aggregate == "mean":
        return {
            "rhythmic_complexity": sum(d["rhythmic_complexity"] for d in per_bar) / bars,
            "pitch_range": sum(d["pitch_range"] for d in per_bar) / bars,
            "note_density": sum(d["note_density"] for d in per_bar) / bars,
            "contour": sum(d["contour"] for d in per_bar) / bars,
        }

    weights_per_bar = []
    for bar_index in range(bars):
        onsets, _ = _quantize_bar_onsets(score, track, bar_index, slots)
        weights_per_bar.append(sum(onsets))

    total_w = sum(weights_per_bar)
    if total_w == 0:
        return {
            "rhythmic_complexity": 0.0,
            "pitch_range": 0.0,
            "note_density": 0.0,
            "contour": 0.0,
        }

    return {
        "rhythmic_complexity": sum(d["rhythmic_complexity"] * w for d, w in zip(per_bar, weights_per_bar)) / total_w,
        "pitch_range": sum(d["pitch_range"] * w for d, w in zip(per_bar, weights_per_bar)) / total_w,
        "note_density": sum(d["note_density"] * w for d, w in zip(per_bar, weights_per_bar)) / total_w,
        "contour": sum(d["contour"] * w for d, w in zip(per_bar, weights_per_bar)) / total_w,
    }
