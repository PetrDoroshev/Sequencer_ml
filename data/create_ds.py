from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from midi_pp import directory_min_max_pitch
from pathlib import Path
from os.path import exists
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from symusic import Score
from tqdm import tqdm
import os
import torch
import shutil
import hashlib
import csv

MAX_PITCH = 109
MIN_PITCH = 21

TOKENIZER_PARAMS = {

    "pitch_range": (MIN_PITCH, MAX_PITCH),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_velocities": False,
    "use_chords": False,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "use_pitch_intervals": True,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
    "remove_duplicated_notes": True
}

class DatasetWithAttributes(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        files_paths: list[Path],
        attributes_map: dict,
        default_num_attributes: int = 1,
    ):
        self.base_dataset = base_dataset
        self.files_paths = [str(Path(p).resolve()) for p in files_paths]
        self.attributes_map = attributes_map
        self.default_num_attributes = default_num_attributes

        if len(self.files_paths) != len(self.base_dataset):
            raise ValueError("files_paths length must match base_dataset length")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        key = self.files_paths[idx]

        attrs = self.attributes_map.get(key)
        if attrs is None:
            attrs = [0.0] * self.default_num_attributes

        item["attributes"] = torch.tensor(attrs, dtype=torch.float32)
        return item


def load_attributes_csv(attributes_csv_path: Path):
    attributes_map = {}

    with open(attributes_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in (reader.fieldnames or []):
            raise ValueError("attributes csv must contain 'path' column")

        attr_columns = [c for c in reader.fieldnames if c != "path"]
        if len(attr_columns) == 0:
            raise ValueError("attributes csv must contain at least one attribute column")

        for row in reader:
            abs_path = str(Path(row["path"]).resolve())
            attrs = [float(row[c]) for c in attr_columns]
            attributes_map[abs_path] = attrs

    return attributes_map, len(attr_columns)


def _collate_with_attributes(base_collator):
    def collate_fn(batch):
        attrs = torch.stack([sample["attributes"] for sample in batch], dim=0)
        base_batch = [{k: v for k, v in sample.items() if k != "attributes"} for sample in batch]
        output = base_collator(base_batch)
        output["attributes"] = attrs
        return output

    return collate_fn


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

def split_files(midis_dir: Path, chunks_dir: Path, tokenizer: REMI, max_seq_len: int):
    
    if not os.path.exists(chunks_dir):
        os.mkdir(chunks_dir)

    midi_paths = list(midis_dir.resolve().glob("**/*.mid"))

    for f in midi_paths:
        print(f)

    split_files_for_training(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        save_dir=chunks_dir,
        max_seq_len=max_seq_len
    )


def rename_files_to_hash(
    midis_dir: Path,
    include_parent: bool = True,
    algorithm: str = "sha1",
    dry_run: bool = False,
):
    if not midis_dir.exists():
        raise FileNotFoundError(midis_dir)

    mapping = {}
    for path in midis_dir.glob("**/*"):
        if not path.is_file():
            continue

        rel = path.relative_to(midis_dir)
        name_for_hash = str(rel) if include_parent else path.name

        h = hashlib.new(algorithm)
        h.update(name_for_hash.encode("utf-8", errors="ignore"))
        new_name = f"{h.hexdigest()}{path.suffix}"
        new_path = path.with_name(new_name)

        if new_path == path:
            continue

        if new_path.exists():
            raise FileExistsError(f"Hash collision or duplicate name: {new_path}")

        mapping[str(path)] = str(new_path)
        if not dry_run:
            path.rename(new_path)

    return mapping

def deduplicate_chunks(chunks_dir: Path, tokenizer: REMI, output_dir: Path = None, copy: bool = True):
    
    if output_dir is None:
        output_dir = chunks_dir.parent / "chunks_dedup"
        
    chunk_files = list(chunks_dir.glob("**/*.mid"))
    print(f"Найдено {len(chunk_files)} чанков.")
        
    unique_chunks = {}
    duplicate_count = 0

    for file_path in tqdm(chunk_files, desc="Обработка чанков"):
       
        score = Score(file_path)
        ids = tokenizer.encode(score)[0].ids
        token_tuple = tuple(ids)

        if token_tuple not in unique_chunks:
            unique_chunks[token_tuple] = file_path
        else:
            duplicate_count += 1

    print(f"Уникальных чанков: {len(unique_chunks)}")
    print(f"Дубликатов: {duplicate_count}")

   
    output_dir.mkdir(parents=True, exist_ok=True)

    for token_tuple, src_path in tqdm(unique_chunks.items(), desc="Сохранение уникальных чанков"):
        dst_path = output_dir / src_path.name
        if copy:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(str(src_path), str(dst_path))
    
    print(f"Уникальные чанки сохранены в: {output_dir}")
    return output_dir

def create_dataloader(
    midis_path: Path,
    tokenizer: REMI,
    subset_fraction=0.3,
    val_split=0.2,
    max_seq_len=512,
    batch_size=32,
    attributes_csv_path: Path = None,
    attributes_map: dict = None,
    default_num_attributes: int = 1,
):

    files_paths = [p.resolve() for p in midis_path.glob("**/*.mid")]

    full_dataset = DatasetMIDI(
        files_paths=files_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )

    with_attributes = False
    loaded_default_num_attributes = default_num_attributes

    if attributes_csv_path is not None:
        attributes_map, loaded_default_num_attributes = load_attributes_csv(attributes_csv_path)
        with_attributes = True

    elif attributes_map is not None:
        with_attributes = True

    if with_attributes:

        full_dataset = DatasetWithAttributes(
            base_dataset=full_dataset,
            files_paths=files_paths,
            attributes_map=attributes_map,
            default_num_attributes=loaded_default_num_attributes,
        )

    total = len(full_dataset)
    subset_size = int(total * subset_fraction)
    indices = torch.randperm(total).tolist()[:subset_size]
    dataset = Subset(full_dataset, indices)
    
    total_size = len(dataset)
    
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    collator = DataCollator(tokenizer.pad_token_id)
    collate_fn = _collate_with_attributes(collator) if with_attributes else collator
    
    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=batch_size)

    return train_dataloader, val_dataloader

def validate_batch(batch, tokenizer: REMI, pad_threshold=0.6):

    input_ids = batch["input_ids"]

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id

    if input_ids.max() >= vocab_size or input_ids.min() < 0:
        raise ValueError("Найдены токены вне диапазона словаря")

    for seq in input_ids:
        if torch.all(seq == pad_id):
            raise ValueError("Найдена полностью PAD последовательность")


    pad_ratio = (input_ids == pad_id).float().mean().item()
    if pad_ratio > pad_threshold:
        print(f"Предупреждение: высокая доля PAD = {pad_ratio:.2f}")


    return pad_ratio


def validate_dataloader(dataloader, tokenizer, max_batches=10):
    
    print("\n Проверка валидности датасета...\n")

    total_pad = 0
    batches_checked = 0

    for i, batch in enumerate(dataloader):
        pad_ratio = validate_batch(batch, tokenizer)
        total_pad += pad_ratio
        batches_checked += 1

        if i >= max_batches:
            break

    print("\nПроверка завершена")
    print(f"Проверено батчей: {batches_checked}")
    print(f"Средняя доля PAD: {total_pad / batches_checked:.3f}")

if __name__ == "__main__":

    MIDIS_DIR = Path("./ds/dMelodies_sample")
    #CHUNKS_DIR = Path("./ds/lakhmidi/chunks")
    #EDUP_CHUNKS_DIR = Path("./ds/lakhmidi/chunks_dedup")    

    if not os.path.exists(Path("./tokenizer/tokenizer.json")):
        tokenizer = create_tokenizer(MIDIS_DIR, Path("./tokenizer"), TOKENIZER_PARAMS)
    else:
        tokenizer = open_tokenizer(Path("./tokenizer"))


    
    dl, _ = create_dataloader(MIDIS_DIR, 
                           tokenizer, subset_fraction=1.0, 
                           val_split=0.2, max_seq_len=128, batch_size=32,
                           attributes_csv_path="./attrs.csv")
    for batch in dl:

        print(batch["attributes"])
        break
       
    
    



    
        




    
