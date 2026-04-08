from typing import List
from symusic import Score
from symusic.core import TrackTick, ScoreTick, TimeSignatureTick
from pathlib import Path
from os.path import exists
from tqdm import tqdm
import os
import math
import mido

def is_midi_valid(filepath: Path) -> bool:
    try:
        mido.MidiFile(filepath)
        return True
    except Exception as e:
        print(f"Ошибка в {filepath}: {e}")
        return False


def directory_min_max_pitch(midis_dir: Path):

    pitches = []
    midis = list(midis_dir.glob("**/*.mid"))

    for midi in midis:

        score = Score(midi)

        for track in score.tracks:
            for note in track.notes:
                pitches.append(note.pitch)

    return min(pitches), max(pitches)

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
    
    if len(track.notes) < 2:
        return False

    notes = sorted(track.notes, key=lambda n: n.time)

    prev_end = notes[0].time + notes[0].duration

    for note in notes[1:]:
        if note.time < prev_end:
            return True
        prev_end = max(prev_end, note.time + note.duration)

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

def trim_track(track: TrackTick) -> TrackTick:
    pass

def is_track_valid(track: TrackTick) -> bool:

    return (not track.is_drum and track.note_num() > 0)
    
if __name__ == "__main__":

    midis_dir = "./ds/vgmusic_midi/full"
    preprocessed_dir = "./ds/vgmusic_midi/preprocessed"

    if not os.path.exists(Path(preprocessed_dir)):
        os.mkdir(Path(preprocessed_dir))

    discarded_count = 0
    total_files = 0
    faulty_files = 0
    
    for filename in tqdm(list(Path(midis_dir).glob("**/*.mid"))):

        #print(f"Processing file: {filename}")
        total_files += 1

        if not is_midi_valid(filename):
            print(f"Invalid midi file: {filename}")
            faulty_files += 1
            continue
        
        try:
            score = Score(filename)
        except Exception as e:
            print(e)
            print(f"Unable to open file: {filename}")
            faulty_files += 1
            continue

        tracks = []

        if len(score.time_signatures) == 0:
            score.time_signatures = [TimeSignatureTick(0, 4, 4)]

        for track in score.tracks:

            if not is_track_valid(track):
                continue

            for splitted_track in split_track(score, track, 1):
                
                if is_track_poly(splitted_track):
                    continue

                bars_count = track_bars_count(score, splitted_track)
                note_density = notes_per_bar(score, splitted_track)

                if bars_count >= 2 and 2 <= note_density <= 32:
                    tracks.append(splitted_track)

        if tracks:
            score.tracks = tracks
            score.dump_midi(Path(preprocessed_dir, f"{filename.parts[-2]}_{filename.name}"))
        else:
            discarded_count += 1
    
    print(f"total_files: {total_files}")
    print(f"faulty_files: {faulty_files}")
    print(f"discarted_files: {discarded_count}")

