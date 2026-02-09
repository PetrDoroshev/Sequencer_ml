from typing import List
from symusic import Score
from symusic.core import TrackTick, ScoreTick, TimeSignatureTick
from pathlib import Path
from os.path import exists
import os

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

def split_track(score: ScoreTick, track: TrackTick, max_bar_gap: int) -> List[TrackTick]:

    splited_tracks = []

    numerator = score.time_signatures[0].numerator
    denominator = score.time_signatures[0].denominator

    bar_tick_duration = score.ticks_per_quarter * 4 * (numerator / denominator)

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

    score = Score(Path("./ds/ctayla.mid"))
    tracks = []

    if len(score.time_signatures) == 0:
        score.time_signatures = [TimeSignatureTick(0, 4, 4)]

    for track in score.tracks:

        if (not is_track_poly(track) and not track.is_drum):
            tracks.append(track)

        
    score.tracks = tracks
    tracks.clear()

    for track in score.tracks:
        for splitted_track in split_track(score, track, 1):
            tracks.append(splitted_track)

    score.tracks = tracks
    score.dump_midi("ctayla_e.mid")

    #preporcess_midi(Path("./ds/nes/train"))