from typing import List
from symusic import Score
from symusic.core import TrackTick, ScoreTick, TimeSignatureTick
from pathlib import Path
from os.path import exists
import os
import math

def track_bars_count(score: ScoreTick, track: TrackTick) -> int:
    
    if track.note_num() == 0:
        return 0

    return math.ceil(track.end() / get_bar_duration(score, track))

def notes_per_bar(score: ScoreTick, track: TrackTick) -> float:

    bars = track_bars_count(score, track)

    if bars == 0:
        return 0

    return (track.note_num() / bars)

def is_track_poly(track: TrackTick) -> bool:
    
    note_start_set = set()

    for note in track.notes:

        if note.time not in note_start_set:
            note_start_set.add(note.time)
        else:
            return True

    return False

def preporcess_midi(midis_dir: Path) -> None:

    if not exists(midis_dir):
        raise FileNotFoundError
    
    for filename in os.listdir(midis_dir):
        score = Score(Path(midis_dir, filename))
        
        for i, track in enumerate(score.tracks):

            if is_track_poly(track):
                print(f"{filename}, track {i} - polyphonic")

            if track.is_drum:
                print(f"{filename}, track {i} - drum")

def get_bar_duration(score: ScoreTick, track: TrackTick) -> int:

    numerator = score.time_signatures[0].numerator
    denominator = score.time_signatures[0].denominator

    bar_tick_duration = score.ticks_per_quarter * numerator * 4 // denominator

    return bar_tick_duration

def split_on_time_signature(score: ScoreTick) -> List[ScoreTick]:

    scores = []
    ts_list = score.time_signatures

    if len(score.time_signatures) < 2:
        return None

    for i, ts in enumerate(ts_list):
        start = ts.time
        end = ts_list[i+1].time if i+1 < len(ts_list) else score.end()

        segment = score.clip(start, end)
        segment.shift_time(-start, inplace=True)
        segment.time_signatures = [TimeSignatureTick(0, ts.numerator, ts.denominator)]

        scores.append(segment)

    return scores

def split_track(score: ScoreTick, track: TrackTick, max_bar_gap: int) -> List[TrackTick]:

    splited_tracks = []

    bar_tick_duration = get_bar_duration(score, track)

    prev_note = track.notes[0]
    clip_start_time = prev_note.time

    for note in track.notes[1:]:

        time_diff = note.time - (prev_note.time + prev_note.duration)
        
        if (time_diff > bar_tick_duration * max_bar_gap):

            new_track = track.clip(clip_start_time, prev_note.time + prev_note.duration)

            if (new_track.notes):

                new_track.shift_time(-new_track.notes[0].time, inplace=True)
                splited_tracks.append(new_track)

            clip_start_time = note.time

        prev_note = note
    
    new_track = track.clip(clip_start_time, prev_note.time + prev_note.duration)
    if (new_track.notes):
        new_track.shift_time(-new_track.notes[0].time, inplace=True)
        splited_tracks.append(new_track)

    return splited_tracks
    
if __name__ == "__main__":

    midis_dir = "./ds/nes"
    preprocessed_dir = "./ds/preprocessed"
    discarded_count = 0

    for filename in os.listdir(Path(midis_dir)):

        print(f"Processing file: {filename}")

        try:
            score = Score(Path(midis_dir, filename))
        except ValueError:
            continue

        tracks = []

        if len(score.time_signatures) == 0:
            score.time_signatures = [TimeSignatureTick(0, 4, 4)]

        for track in score.tracks:

            if (not is_track_poly(track) and 
                not track.is_drum and
                    track.note_num() > 0):

                tracks.append(track)

        score.tracks = tracks
        tracks.clear()

        for track in score.tracks:
            for splitted_track in split_track(score, track, 1):
                tracks.append(splitted_track)

        score.tracks = tracks
        tracks.clear()

        for track in score.tracks:

            if 4 <= track_bars_count(score, track) <= 16 and 4 <= notes_per_bar(score, track) <= 16:
                tracks.append(track)

        score.tracks = tracks

        if score.tracks:
            score.tracks = tracks
            score.dump_midi(Path(preprocessed_dir, filename))
        else:
            discarded_count += 1
        
    print(f"discarted_files: {discarded_count}")

