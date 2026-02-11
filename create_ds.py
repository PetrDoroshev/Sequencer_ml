from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from pathlib import Path
from os.path import exists
from torch.utils.data import DataLoader
from symusic import Score
import os

TOKENIZER_PARAMS = {

    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "special_tokens": [],
    "use_velocities": False,
    "use_chords": False,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}

def create_tokenizer(midis_dir: Path, tokenizer_dir: Path, tokenizer_params: dict) -> REMI:

    config = TokenizerConfig(**tokenizer_params)
    tokenizer = REMI(config)

    root_dir = midis_dir.resolve()
    midi_paths = list(root_dir.glob("**/*.mid"))

    #tokenizer.train(vocab_size=30000, files_paths=midi_paths)
    tokenizer.save_pretrained(tokenizer_dir)

    return tokenizer

def open_tokenizer(tokenizer_dir: Path) -> REMI:

    if not exists(Path(tokenizer_dir, "tokenizer.json")):
        return FileNotFoundError

    tokenizer = REMI.from_pretrained(tokenizer_dir, local_files_only=True)
    return tokenizer


def create_dataloader(midis_path: Path, tokenizer: REMI) -> DataLoader:

    dataset_chunks_dir = Path(str(midis_path), "chunks")

    if len(os.listdir(Path(dataset_chunks_dir))) <= 1:
        split_files_for_training(
            files_paths=list(midis_path.resolve().glob("**/*.mid")),
            tokenizer=tokenizer,
            save_dir=dataset_chunks_dir,
            max_seq_len=512
        )

    dataset = DatasetMIDI(
        files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=512
    )

    collator = DataCollator(tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=32)

    return data_loader

if __name__ == "__main__":

    
    if not exists(Path("tokenizer/tokenizer.json")):
        tokenizer = create_tokenizer(Path("./ds/preprocessed"), Path("./tokenizer"), TOKENIZER_PARAMS)

    else:
        tokenizer = open_tokenizer(Path("./tokenizer"))

    
    data_loader = create_dataloader(Path("./ds/preprocessed"), tokenizer)
 
    for batch in data_loader:
        ids = batch["input_ids"][0].tolist()
        print(len(ids))
        #print(ids.count(0))



        
    
    

    

    