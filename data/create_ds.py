from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
#from midi_pp import directory_min_max_pitch
from pathlib import Path
from os.path import exists
from torch.utils.data import DataLoader, random_split, Dataset, Subset, BatchSampler
from symusic import Score
from tqdm import tqdm
import os
import torch
import shutil
import hashlib
import csv
import math

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


class StratifiedBatchSampler(BatchSampler):
    
    def __init__(self, strata_labels: list[int], batch_size: int, drop_last: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(strata_labels) == 0:
            raise ValueError("strata_labels must be non-empty")

        self.strata_labels = list(strata_labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self._indices_by_strata = {}
        for idx, stratum in enumerate(self.strata_labels):
            self._indices_by_strata.setdefault(int(stratum), []).append(idx)
        self._strata = sorted(self._indices_by_strata.keys())

    def __len__(self):
        n = len(self.strata_labels)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        # Per-epoch shuffle inside each stratum.
        pools = {}
        for stratum, indices in self._indices_by_strata.items():
            perm = torch.randperm(len(indices)).tolist()
            pools[stratum] = [indices[i] for i in perm]

        for _ in range(len(self)):
            active = [s for s in self._strata if len(pools[s]) > 0]
            if len(active) == 0:
                break

            active_perm = torch.randperm(len(active)).tolist()
            active = [active[i] for i in active_perm]
            batch = []

            # 1) Ensure representation from all active strata when possible.
            target_per_stratum = max(1, self.batch_size // len(active))
            for stratum in active:
                take = min(target_per_stratum, len(pools[stratum]), self.batch_size - len(batch))
                for _ in range(take):
                    batch.append(pools[stratum].pop())
                if len(batch) == self.batch_size:
                    break

            # 2) Fill remaining slots from any non-empty stratum.
            while len(batch) < self.batch_size:
                active_non_empty = [s for s in active if len(pools[s]) > 0]
                if len(active_non_empty) == 0:
                    break
                for stratum in active_non_empty:
                    if len(batch) == self.batch_size:
                        break
                    batch.append(pools[stratum].pop())

            if self.drop_last and len(batch) < self.batch_size:
                continue

            if len(batch) > 0:
                yield batch


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


def _bucketize_by_quantiles(values: torch.Tensor, num_bins: int) -> torch.Tensor:
    if values.numel() == 0 or num_bins <= 1:
        return torch.zeros_like(values, dtype=torch.long)

    quantiles = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=values.device)
    edges = torch.quantile(values, quantiles)

    # Keep only unique internal boundaries; duplicates appear on low-variance data.
    unique_boundaries = []
    for boundary in edges[1:-1].tolist():
        if len(unique_boundaries) == 0 or boundary > unique_boundaries[-1]:
            unique_boundaries.append(boundary)

    if len(unique_boundaries) == 0:
        return torch.zeros_like(values, dtype=torch.long)

    boundaries = torch.tensor(unique_boundaries, dtype=values.dtype, device=values.device)
    return torch.bucketize(values, boundaries, right=False)


def _resolve_subset_to_base_indices(dataset: Dataset):
    if isinstance(dataset, Subset):
        parent = _resolve_subset_to_base_indices(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    return list(range(len(dataset)))


def _build_rarity_strata_labels(
    dataset: Dataset,
    freq_num_bins: int,
    strata_num_bins: int,
):
    # Works only when the base dataset contains attribute metadata.
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset

    if not isinstance(base, DatasetWithAttributes):
        return None

    base_indices = _resolve_subset_to_base_indices(dataset)
    if len(base_indices) == 0:
        return None

    attrs_rows = []
    for base_idx in base_indices:
        key = base.files_paths[base_idx]
        attrs = base.attributes_map.get(key)
        if attrs is None:
            attrs = [0.0] * base.default_num_attributes
        attrs_rows.append(attrs)

    attrs_tensor = torch.tensor(attrs_rows, dtype=torch.float32)
    if attrs_tensor.ndim != 2 or attrs_tensor.size(0) == 0:
        return None

    n_samples, n_attrs = attrs_tensor.shape
    rarity_score = torch.zeros(n_samples, dtype=torch.float32)

    for col in range(n_attrs):
        bins = _bucketize_by_quantiles(attrs_tensor[:, col], freq_num_bins)
        counts = torch.bincount(bins, minlength=int(bins.max().item()) + 1).float()
        rarity_score += 1.0 / counts[bins].clamp_min(1.0)

    rarity_score = rarity_score / max(n_attrs, 1)
    strata = _bucketize_by_quantiles(rarity_score, strata_num_bins)
    return strata.tolist()


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

def create_dataset(
    midis_path: Path,
    tokenizer: REMI,
    attributes_csv_path: Path,
    subset_fraction=0.3,
    val_split=0.2,
    test_split=0.1,
    max_seq_len=512,
    
):
    files_paths = [p.resolve() for p in midis_path.glob("**/*.mid")]

    full_dataset = DatasetMIDI(
        files_paths=files_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )

    attributes_map, loaded_default_num_attributes = load_attributes_csv(attributes_csv_path)
    
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
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, 
                                                                      val_size, 
                                                                      test_size])
    
    return train_dataset, val_dataset, test_dataset
    
def create_dataloader(
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizer: REMI,
    max_seq_len=512,
    batch_size=32,
    use_stratified_sampler: bool = True,
    frequency_bins: int = 8,
    strata_bins: int = 3,
):

    collator = DataCollator(tokenizer.pad_token_id)
    collate_fn = _collate_with_attributes(collator)

    train_batch_sampler = None
    if use_stratified_sampler:
        strata_labels = _build_rarity_strata_labels(
            train_dataset,
            freq_num_bins=frequency_bins,
            strata_num_bins=strata_bins,
        )
        if strata_labels is not None and len(set(strata_labels)) > 1:
            train_batch_sampler = StratifiedBatchSampler(
                strata_labels=strata_labels,
                batch_size=batch_size,
                drop_last=False,
            )

    if train_batch_sampler is not None:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=collate_fn,
            batch_sampler=train_batch_sampler,
        )
    else:
        train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=batch_size)

    val_dataloader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, collate_fn=collate_fn, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader

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

    MIDIS_DIR = Path("./ds/vgmusic_lakh_folk/preprocessed")
    CHUNKS_DIR = Path("./ds/vgmusic_lakh_folk/chunks")
    DEDUP_CHUNKS_DIR = Path("./ds/vgmusic_lakh_folk/chunks_dedup")  

    MIN_PITCH, MAX_PITCH = directory_min_max_pitch(MIDIS_DIR)

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

    if not os.path.exists(Path("./tokenizer/tokenizer.json")):
        tokenizer = create_tokenizer(MIDIS_DIR, Path("./tokenizer"), TOKENIZER_PARAMS)
    else:
        tokenizer = open_tokenizer(Path("./tokenizer"))

    
    split_files(MIDIS_DIR, CHUNKS_DIR, tokenizer, 512)
    deduplicate_chunks(CHUNKS_DIR, tokenizer, DEDUP_CHUNKS_DIR)

    
    



    
        




    
