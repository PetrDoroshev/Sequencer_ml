from music21 import converter
from pathlib import Path
import os
import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

def abc_file_to_midi(abc_file_path, output_midi_path):
    
    try:
        score = converter.parse(abc_file_path)
        score.write('midi', fp=output_midi_path)

    except Exception as e:
        pass


if __name__ == "__main__":

    files = list(Path("./ds/arvae_data/folk_raw_data").glob("**/*.abc"))
    files.sort()

    for filename in files:
        
        midi = Path(f"./ds/arvae_data/midi/{filename.stem}.mid")
        if not os.path.exists(midi.resolve()):

            signal.alarm(300)
            print(f"Processing file {filename}...")

            try:
                abc_file_to_midi(str(filename.resolve()), str(midi))
            except TimeoutException:
                continue 
            else:
                signal.alarm(0)
        else:
            print(f"File {filename} has been already processed")