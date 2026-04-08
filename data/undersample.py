from __future__ import annotations
import pandas as pd
import numpy as np
import shutil
import hashlib
from pathlib import Path
from typing import List, Union, Optional, Dict
import matplotlib.pyplot as plt
from symusic import Score
from miditok.data_augmentation import augment_dataset
from attrubute import plot_attributes


ATTRIBUTE_COLUMNS = ['pitch_range', 'note_density', 'contour']
DEFAULT_FIXED_BIN_RANGES = {
    'pitch_range': (0.0, 1.0),
    'note_density': (0.0, 1.0),
    'contour': (-1.0, 1.0),
}

def plot_strata_distribution(
    df: pd.DataFrame,
    bins_per_attr: int = 5,
    title: str = "Распределение по стратам",
    ax: plt.Axes = None,
    show_labels: bool = True
):
    """
    Строит гистограмму частот страт, где лейблы отображают диапазоны значений атрибутов.
    """
    # Вычисляем бины (квантильные) для каждого атрибута
    bins = {}
    for attr in ATTRIBUTE_COLUMNS:
        percentiles = np.linspace(0, 100, bins_per_attr + 1)[1:-1]
        edges = np.percentile(df[attr], percentiles)
        bins[attr] = np.concatenate([[-np.inf], edges, [np.inf]])

    # Функция для получения интервала в читаемом виде
    def format_interval(val, edges, idx):
        low = edges[idx]
        high = edges[idx+1]
        if low == -np.inf:
            return f"(-∞, {high:.2f})"
        elif high == np.inf:
            return f"[{low:.2f}, +∞)"
        else:
            return f"[{low:.2f}, {high:.2f})"

    # Назначаем бины и создаём метки страт
    strata_labels = []
    for _, row in df.iterrows():
        labels = []
        for attr in ATTRIBUTE_COLUMNS:
            val = row[attr]
            edges = bins[attr]
            bin_idx = np.digitize(val, edges[1:-1])  # 0..bins_per_attr-1
            interval = format_interval(val, edges, bin_idx)
            labels.append(f"{attr}: {interval}")
        strata_labels.append(' | '.join(labels))

    # Строим гистограмму
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    # Для читаемости группируем по уникальным меткам и сортируем
    unique, counts = np.unique(strata_labels, return_counts=True)
    # Сортируем по частоте для наглядности
    sort_idx = np.argsort(counts)[::-1]
    unique = unique[sort_idx]
    counts = counts[sort_idx]
    # Показываем только первые N (чтобы не перегружать график)
    if len(unique) > 30:
        unique = unique[:30]
        counts = counts[:30]
    bars = ax.bar(range(len(unique)), counts)
    ax.set_xticks(range(len(unique)))
    if show_labels:
        ax.set_xticklabels(unique, rotation=90, fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("Количество примеров")
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    return ax


def _build_quantile_bins(df: pd.DataFrame, bins_per_attr: int):
    bins = {}
    for attr in ATTRIBUTE_COLUMNS:
        percentiles = np.linspace(0, 100, bins_per_attr + 1)[1:-1]
        edges = np.percentile(df[attr], percentiles)
        bins[attr] = np.concatenate([[-np.inf], edges, [np.inf]])
    return bins


def _assign_bin(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges[1:-1])


def _build_fixed_bins(
    bins_per_attr: int,
    fixed_bin_ranges: Optional[Dict[str, tuple[float, float]]] = None,
):
    fixed_bin_ranges = fixed_bin_ranges or DEFAULT_FIXED_BIN_RANGES
    bins = {}
    for attr in ATTRIBUTE_COLUMNS:
        low, high = fixed_bin_ranges.get(attr, DEFAULT_FIXED_BIN_RANGES[attr])
        if high <= low:
            raise ValueError(f"Для {attr} правая граница должна быть больше левой")
        edges = np.linspace(low, high, bins_per_attr + 1)[1:-1]
        bins[attr] = np.concatenate([[-np.inf], edges, [np.inf]])
    return bins


def _prepare_binned_dataframe(
    df: pd.DataFrame,
    bins_per_attr: int,
    binning_mode: str = "quantile",
    fixed_bin_ranges: Optional[Dict[str, tuple[float, float]]] = None,
) -> pd.DataFrame:
    if binning_mode == "quantile":
        bins = _build_quantile_bins(df, bins_per_attr)
    elif binning_mode == "fixed":
        bins = _build_fixed_bins(bins_per_attr, fixed_bin_ranges)
    else:
        raise ValueError("binning_mode должен быть 'quantile' или 'fixed'")

    prepared_df = df.copy()

    for attr in ATTRIBUTE_COLUMNS:
        prepared_df[f"{attr}_bin"] = _assign_bin(prepared_df[attr], bins[attr])

    prepared_df['stratum'] = (
        prepared_df['pitch_range_bin'].astype(str) + '_' +
        prepared_df['note_density_bin'].astype(str) + '_' +
        prepared_df['contour_bin'].astype(str)
    )
    return prepared_df


def _resolve_midi_path(path_value: Union[str, Path], midi_dir: Path, csv_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    midi_candidate = midi_dir / path
    if midi_candidate.exists():
        return midi_candidate.resolve()

    csv_candidate = csv_path.parent / path
    if csv_candidate.exists():
        return csv_candidate.resolve()

    return midi_candidate.resolve()


def _get_valid_transpose_shifts(score: Score, max_abs_semitones: int) -> List[int]:
    pitches = [
        note.pitch
        for track in score.tracks
        for note in track.notes
    ]
    if not pitches:
        return []

    min_pitch = min(pitches)
    max_pitch = max(pitches)

    valid_shifts = []
    for shift in range(-max_abs_semitones, max_abs_semitones + 1):
        if shift == 0:
            continue
        if 0 <= min_pitch + shift and max_pitch + shift <= 127:
            valid_shifts.append(shift)
    return valid_shifts


def _transpose_score_inplace(score: Score, semitones: int) -> None:
    for track in score.tracks:
        for note in track.notes:
            note.pitch += semitones


def _resolve_target_size(counts: pd.Series, explicit_target: Optional[int], use_max: bool) -> int:
    if explicit_target is not None:
        return int(explicit_target)
    if counts.empty:
        return 0
    return int(counts.max() if use_max else counts.min())


def _sample_rows(group: pd.DataFrame, keep_count: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(group) <= keep_count:
        return group
    chosen = rng.choice(group.index.to_numpy(), size=keep_count, replace=False)
    return group.loc[chosen]


def _drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    helper_cols = ['stratum'] + [f"{attr}_bin" for attr in ATTRIBUTE_COLUMNS]
    return df.drop(columns=helper_cols, errors='ignore')


def _build_augmented_filename(
    source_path: Path,
    semitones: int,
    duplicate_id: int,
    max_stem_length: int = 48,
) -> str:
    short_stem = source_path.stem[:max_stem_length]
    source_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{short_stem}_{source_hash}_tr{semitones:+d}_{duplicate_id}{source_path.suffix}"


def thin_contour_tail_and_rescale(
    csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
    attribute: str = "contour",
    soft_threshold: float = 0.20,
    hard_threshold: float = 0.35,
    keep_mid_fraction: float = 0.4,
    keep_high_fraction: float = 0.2,
    random_state: int = 42,
    clip_abs_quantile: Optional[float] = 0.99,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Симметрично прорежает хвосты signed-атрибута по абсолютному значению и
    затем симметрично нормализует его в [0, 1], сохраняя 0 -> 0.5.

    Логика:
    - |x| <= soft_threshold: оставить все;
    - soft_threshold < |x| <= hard_threshold: оставить keep_mid_fraction;
    - |x| > hard_threshold: оставить keep_high_fraction.

    После отбора значения клиппируются по симметричной границе и
    преобразуются по формуле:
        x_norm = (x + bound) / (2 * bound)
    где bound > 0 и x=0 всегда переходит в 0.5.
    """
    csv_path = Path(csv_path)
    output_csv_path = Path(output_csv_path)

    df = pd.read_csv(csv_path)
    if attribute not in df.columns:
        raise ValueError(f"CSV должен содержать колонку '{attribute}'")
    if hard_threshold < soft_threshold:
        raise ValueError("hard_threshold должен быть >= soft_threshold")
    if not (0.0 <= keep_mid_fraction <= 1.0):
        raise ValueError("keep_mid_fraction должен быть в [0, 1]")
    if not (0.0 <= keep_high_fraction <= 1.0):
        raise ValueError("keep_high_fraction должен быть в [0, 1]")
    if clip_abs_quantile is not None and not (0.0 < clip_abs_quantile <= 1.0):
        raise ValueError("clip_abs_quantile должен быть в (0, 1]")

    rng = np.random.default_rng(random_state)
    values = df[attribute].to_numpy(dtype=np.float32)
    abs_values = np.abs(values)

    low_mask = abs_values <= soft_threshold
    mid_mask = (abs_values > soft_threshold) & (abs_values <= hard_threshold)
    high_mask = abs_values > hard_threshold

    keep_mask = np.zeros(len(df), dtype=bool)
    keep_mask[low_mask] = True
    keep_mask[mid_mask] = rng.random(np.sum(mid_mask)) < keep_mid_fraction
    keep_mask[high_mask] = rng.random(np.sum(high_mask)) < keep_high_fraction

    filtered_df = df.loc[keep_mask].copy()
    filtered_values = filtered_df[attribute].to_numpy(dtype=np.float32)

    if filtered_values.size == 0:
        raise ValueError("После прореживания не осталось ни одной строки")

    if clip_abs_quantile is None:
        bound = float(np.max(np.abs(filtered_values)))
    else:
        bound = float(np.quantile(np.abs(filtered_values), clip_abs_quantile))

    if bound <= 1e-12:
        filtered_df[attribute] = 0.5
    else:
        clipped = np.clip(filtered_values, -bound, bound)
        filtered_df[attribute] = (clipped + bound) / (2.0 * bound)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv_path, index=False)

    if verbose:
        print(
            f"{attribute}: kept {len(filtered_df)} / {len(df)} rows | "
            f"soft={soft_threshold:.3f} hard={hard_threshold:.3f} "
            f"mid_keep={keep_mid_fraction:.2f} high_keep={keep_high_fraction:.2f} "
            f"bound={bound:.4f}"
        )

    return filtered_df


def save_midi_files_from_csv(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    midi_dir: Optional[Union[str, Path]] = None,
    copy_files: bool = True,
    verbose: bool = True,
) -> List[Path]:
    """
    Копирует или перемещает MIDI-файлы, перечисленные в CSV с колонкой `path`.
    Возвращает список сохранённых путей.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = Path(midi_dir) if midi_dir is not None else csv_path.parent

    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV должен содержать колонку 'path'")

    saved_paths = []
    for _, row in df.iterrows():
        src = _resolve_midi_path(row["path"], midi_dir, csv_path)
        if not src.exists():
            if verbose:
                print(f"Файл не найден: {src}, пропускаем")
            continue

        dst = output_dir / src.name
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = output_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        if copy_files:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

        saved_paths.append(dst)

    if verbose:
        print(f"Сохранено {len(saved_paths)} MIDI-файлов в {output_dir}")

    return saved_paths


def oversample_by_attributes(
    midi_dir: Union[str, Path],
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    bins_per_attr: int = 5,
    target_per_stratum: Optional[int] = None,
    max_abs_semitones: int = 6,
    random_state: int = 42,
    rarity_mode: str = "joint",
    binning_mode: str = "quantile",
    fixed_bin_ranges: Optional[Dict[str, tuple[float, float]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Выполняет оверсемплинг редких страт датасета и для добавленных примеров
    создаёт транспонированные копии MIDI-файлов.

    В режиме `joint` редкость определяется по комбинации квантильных бинов
    атрибутов `pitch_range`, `note_density`, `contour`.
    В режиме `per_attribute` каждый атрибут балансируется независимо.

    Параметры
    ----------
    midi_dir : Path или str
        Базовая директория с MIDI-файлами.
    csv_path : Path или str
        Путь к CSV c колонками: path, pitch_range, note_density, contour.
    output_dir : Path или str, optional
        Куда сохранить итоговый датасет. Если не задан, создаётся папка
        `oversampled` рядом с CSV.
    bins_per_attr : int, default=5
        Количество квантильных бинов на каждый атрибут.
    target_per_stratum : int, optional
        Целевое количество файлов в каждой страте. Если не задано,
        используется размер самой крупной страты.
    max_abs_semitones : int, default=6
        Максимальное смещение при транспонировании в полутонах.
    random_state : int, default=42
        Seed для воспроизводимости.
    rarity_mode : {"joint", "per_attribute"}, default="joint"
        Как определять редкость: по совместной страте или отдельно
        для каждого атрибута.
    binning_mode : {"quantile", "fixed"}, default="quantile"
        Как строить бины: по квантилям или по фиксированным диапазонам.
    fixed_bin_ranges : dict, optional
        Диапазоны для fixed-бинов в формате
        {"pitch_range": (0, 1), "note_density": (0, 1), "contour": (-1, 1)}.
    verbose : bool, default=True
        Печатать ход обработки.

    Возвращает
    ----------
    pd.DataFrame
        DataFrame итогового датасета с обновлёнными путями.
    """
    midi_dir = Path(midi_dir)
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / "oversampled"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).copy()
    required_cols = ['path', *ATTRIBUTE_COLUMNS]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV должен содержать колонки: {', '.join(required_cols)}")
    if rarity_mode not in {"joint", "per_attribute"}:
        raise ValueError("rarity_mode должен быть 'joint' или 'per_attribute'")
    if binning_mode not in {"quantile", "fixed"}:
        raise ValueError("binning_mode должен быть 'quantile' или 'fixed'")

    rng = np.random.default_rng(random_state)
    df = _prepare_binned_dataframe(df, bins_per_attr, binning_mode, fixed_bin_ranges)

    if rarity_mode == "joint":
        counts = df['stratum'].value_counts().sort_index()
    else:
        counts = pd.concat(
            [df[f"{attr}_bin"].value_counts().rename(attr) for attr in ATTRIBUTE_COLUMNS],
            axis=1,
        ).fillna(0).astype(int)

    target_per_stratum = _resolve_target_size(
        df['stratum'].value_counts() if rarity_mode == "joint" else counts.stack(),
        target_per_stratum,
        use_max=True,
    )

    if verbose:
        print(f"Всего файлов в исходном датасете: {len(df)}")
        if rarity_mode == "joint":
            print(f"Всего страт: {df['stratum'].nunique()}")
            print(f"Целевой размер каждой страты: {target_per_stratum}")
        else:
            print(f"Режим редкости: отдельно по каждому атрибуту")
            print(f"Целевой размер каждого бина: {target_per_stratum}")
        print(f"Режим биннинга: {binning_mode}")

    result_rows = []
    created_files = 0
    skipped_files = 0

    for _, row in df.iterrows():
        src = _resolve_midi_path(row['path'], midi_dir, csv_path)
        if not src.exists():
            skipped_files += 1
            if verbose:
                print(f"Файл не найден, пропускаем: {src}")
            continue

        dst = output_dir / src.name
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)

        row_copy = row.copy()
        row_copy['path'] = str(dst.resolve())
        result_rows.append(row_copy)

    valid_df = pd.DataFrame(result_rows)
    if valid_df.empty:
        raise ValueError("Не удалось найти ни одного MIDI-файла из CSV.")

    current_df = valid_df.copy()

    def create_augmented_copy(source_row: pd.Series, duplicate_id: int) -> Optional[pd.Series]:
        nonlocal created_files, skipped_files

        source_path = Path(source_row['path'])
        score = Score(source_path)
        valid_shifts = _get_valid_transpose_shifts(score, max_abs_semitones)
        if not valid_shifts:
            skipped_files += 1
            if verbose:
                print(f"Не удалось транспонировать {source_path.name}: нет допустимых смещений")
            return None

        semitones = int(rng.choice(valid_shifts))
        _transpose_score_inplace(score, semitones)

        new_name = _build_augmented_filename(source_path, semitones, duplicate_id)
        dst = output_dir / new_name
        while dst.exists():
            duplicate_id += 1
            new_name = _build_augmented_filename(source_path, semitones, duplicate_id)
            dst = output_dir / new_name

        score.dump_midi(dst)

        new_row = source_row.copy()
        new_row['path'] = str(dst.resolve())
        created_files += 1
        return new_row

    if rarity_mode == "joint":
        strata_counts = current_df['stratum'].value_counts().sort_index()
        for stratum, count in strata_counts.items():
            needed = max(0, target_per_stratum - int(count))
            if needed == 0:
                continue

            stratum_rows = current_df[current_df['stratum'] == stratum]
            if stratum_rows.empty:
                continue

            row_indices = stratum_rows.index.to_list()
            for extra_idx in range(needed):
                source_row = current_df.loc[int(rng.choice(row_indices))]
                new_row = create_augmented_copy(source_row, extra_idx + 1)
                if new_row is not None:
                    current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        duplicate_id = 1
        blocked_candidates = set()
        while True:
            deficits = []
            for attr in ATTRIBUTE_COLUMNS:
                attr_counts = current_df[f"{attr}_bin"].value_counts()
                for bin_id, count in attr_counts.items():
                    deficit = target_per_stratum - int(count)
                    if deficit > 0:
                        deficits.append((deficit, attr, int(bin_id)))

            if not deficits:
                break

            deficits.sort(reverse=True)
            _, attr, bin_id = deficits[0]
            candidate_rows = current_df[current_df[f"{attr}_bin"] == bin_id]
            if candidate_rows.empty:
                break

            available_rows = candidate_rows[
                ~candidate_rows['path'].isin({
                    path for blocked_attr, blocked_bin, path in blocked_candidates
                    if blocked_attr == attr and blocked_bin == bin_id
                })
            ]
            if available_rows.empty:
                skipped_files += 1
                if verbose:
                    print(f"Для {attr}_bin={bin_id} не осталось файлов, пригодных для транспонирования")
                break

            source_row = available_rows.sample(
                n=1,
                replace=True,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).iloc[0]
            new_row = create_augmented_copy(source_row, duplicate_id)
            duplicate_id += 1
            if new_row is None:
                blocked_candidates.add((attr, bin_id, source_row['path']))
                continue

            current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

    result_df = _drop_helper_columns(current_df)
    result_csv_path = output_dir / "attributes_oversampled.csv"
    result_df.to_csv(result_csv_path, index=False)

    if verbose:
        print(f"Создано транспонированных файлов: {created_files}")
        print(f"Пропущено файлов: {skipped_files}")
        print(f"Итоговый размер датасета: {len(result_df)}")
        print(f"CSV сохранён в: {result_csv_path}")

    return result_df



def undersample_by_attributes(
    midi_dir: Union[str, Path],
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    bins_per_attr: int = 5,
    max_per_stratum: Optional[int] = None,
    target_strata_fraction: Optional[float] = None,
    random_state: int = 42,
    copy_files: bool = True,
    rarity_mode: str = "joint",
    binning_mode: str = "quantile",
    fixed_bin_ranges: Optional[Dict[str, tuple[float, float]]] = None,
    verbose: bool = True
) -> List[Path]:
    """
    Уменьшает количество MIDI‑файлов в датасете, балансируя распределение
    по трём атрибутам: pitch_range, note_density, contour.

    Параметры
    ----------
    midi_dir : Path или str
        Папка, содержащая MIDI‑файлы (пути в CSV должны быть относительными или абсолютными).
    csv_path : Path или str
        Путь к CSV‑файлу с колонками: path, pitch_range, note_density, contour.
    output_dir : Path или str, опционально
        Папка, куда будут скопированы/перемещены отобранные файлы.
        Если не задан, создаётся папка `undersampled` рядом с `csv_path`.
    bins_per_attr : int, default=5
        Количество бинов (квантилей) для каждого атрибута.
    max_per_stratum : int, опционально
        Максимальное количество файлов, сохраняемых из одной страты (комбинации бинов).
        Если не задан, берётся минимальный размер страты (чтобы выровнять все).
    target_strata_fraction : float, опционально
        Альтернатива `max_per_stratum`: доля от исходного размера страты,
        которую нужно оставить (например, 0.2). Если указан и max_per_stratum тоже, приоритет у max_per_stratum.
    random_state : int, default=42
        Для воспроизводимости случайного выбора внутри страт.
    copy_files : bool, default=True
        Если True, файлы копируются; если False – перемещаются (удаляются из исходной папки).
    rarity_mode : {"joint", "per_attribute"}, default="joint"
        Как определять редкость: по совместной страте или отдельно
        для каждого атрибута.
    binning_mode : {"quantile", "fixed"}, default="quantile"
        Как строить бины: по квантилям или по фиксированным диапазонам.
    fixed_bin_ranges : dict, optional
        Диапазоны для fixed-бинов в формате
        {"pitch_range": (0, 1), "note_density": (0, 1), "contour": (-1, 1)}.
    verbose : bool, default=True
        Печатать информацию о процессе.

    Возвращает
    ----------
    List[Path]
        Список путей к отобранным файлам (в output_dir, если задан, иначе исходные).
    """
    # Преобразуем пути
    midi_dir = Path(midi_dir)
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / "undersampled"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем CSV
    df = pd.read_csv(csv_path)
    required_cols = ['path', *ATTRIBUTE_COLUMNS]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV должен содержать колонку '{col}'")
    if rarity_mode not in {"joint", "per_attribute"}:
        raise ValueError("rarity_mode должен быть 'joint' или 'per_attribute'")
    if binning_mode not in {"quantile", "fixed"}:
        raise ValueError("binning_mode должен быть 'quantile' или 'fixed'")

    rng = np.random.default_rng(random_state)
    df = _prepare_binned_dataframe(df, bins_per_attr, binning_mode, fixed_bin_ranges)

    if rarity_mode == "joint":
        counts = df['stratum'].value_counts()
        if verbose:
            print(f"Всего уникальных страт: {len(counts)}")
            print(f"Минимальный размер страты: {counts.min()}")
            print(f"Максимальный размер страты: {counts.max()}")
            print(f"Средний размер страты: {counts.mean():.1f}")

        if max_per_stratum is None:
            if target_strata_fraction is not None:
                max_per_stratum = int(counts.min() / target_strata_fraction)
            else:
                max_per_stratum = int(counts.min())
        if verbose:
            print(f"Оставляем не более {max_per_stratum} примеров на страту")
            print(f"Режим биннинга: {binning_mode}")

        selected_parts = []
        for _, group in df.groupby('stratum'):
            selected_parts.append(_sample_rows(group, max_per_stratum, rng))
        selected_df = pd.concat(selected_parts, ignore_index=False).copy()
    else:
        counts_series = []
        for attr in ATTRIBUTE_COLUMNS:
            attr_counts = df[f"{attr}_bin"].value_counts()
            counts_series.append(attr_counts)
            if verbose:
                print(
                    f"{attr}: бинов={len(attr_counts)}, min={attr_counts.min()}, "
                    f"max={attr_counts.max()}, mean={attr_counts.mean():.1f}"
                )

        all_counts = pd.concat(counts_series)
        if max_per_stratum is None:
            if target_strata_fraction is not None:
                max_per_stratum = int(all_counts.min() / target_strata_fraction)
            else:
                max_per_stratum = int(all_counts.min())
        if verbose:
            print(f"Оставляем не более {max_per_stratum} примеров в каждом бине каждого атрибута")
            print(f"Режим биннинга: {binning_mode}")

        selected_df = df.copy()
        for attr in ATTRIBUTE_COLUMNS:
            filtered_parts = []
            for _, group in selected_df.groupby(f"{attr}_bin"):
                filtered_parts.append(_sample_rows(group, max_per_stratum, rng))
            selected_df = pd.concat(filtered_parts, ignore_index=False).copy()

    selected_df = selected_df.sort_index()
    if verbose:
        print(f"Отобрано {len(selected_df)} файлов из {len(df)}")

    # Обрабатываем файлы
    selected_paths = []
    output_rows = []
    for _, row in selected_df.iterrows():
        src = _resolve_midi_path(row['path'], midi_dir, csv_path)
        if not src.exists():
            if verbose:
                print(f"Файл не найден: {src}, пропускаем")
            continue
        dst = output_dir / src.name
        # Если имя конфликтует, добавим суффикс (простой вариант)
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = output_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        if copy_files:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        selected_paths.append(dst)
        row_copy = row.copy()
        row_copy['path'] = str(dst)
        output_rows.append(row_copy)

    # Сохраняем обновлённый CSV
    result_df = _drop_helper_columns(pd.DataFrame(output_rows))
    result_df.to_csv(output_dir / "attributes_undersampled.csv", index=False)
    if verbose:
        print(f"Готово. Результаты в {output_dir}")

    return selected_paths



def thin_tail_and_rescale(
    csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
    attribute: str,
    soft_threshold: float,
    hard_threshold: float,
    keep_mid_fraction: float = 0.4,
    keep_high_fraction: float = 0.2,
    random_state: int = 42,
    clip_quantiles: tuple[float, float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Прорежает верхний хвост одного атрибута и затем нормализует его в [0, 1].

    Логика:
    - x <= soft_threshold: оставить все
    - soft_threshold < x <= hard_threshold: оставить keep_mid_fraction
    - x > hard_threshold: оставить keep_high_fraction
    - после отбора выполнить min-max нормализацию атрибута в [0, 1]

    Если clip_quantiles заданы, перед min-max делается дополнительный clip
    по квантилям уже на отфильтрованном датасете.
    """
    csv_path = Path(csv_path)
    output_csv_path = Path(output_csv_path)

    df = pd.read_csv(csv_path)
    if attribute not in df.columns:
        raise ValueError(f"Column '{attribute}' not found in {csv_path}")

    if not (0.0 <= keep_mid_fraction <= 1.0):
        raise ValueError("keep_mid_fraction must be in [0, 1]")
    if not (0.0 <= keep_high_fraction <= 1.0):
        raise ValueError("keep_high_fraction must be in [0, 1]")
    if hard_threshold < soft_threshold:
        raise ValueError("hard_threshold must be >= soft_threshold")

    rng = np.random.default_rng(random_state)

    values = df[attribute].to_numpy()

    low_mask = values <= soft_threshold
    mid_mask = (values > soft_threshold) & (values <= hard_threshold)
    high_mask = values > hard_threshold

    keep_mask = np.zeros(len(df), dtype=bool)
    keep_mask[low_mask] = True
    keep_mask[mid_mask] = rng.random(np.sum(mid_mask)) < keep_mid_fraction
    keep_mask[high_mask] = rng.random(np.sum(high_mask)) < keep_high_fraction

    filtered_df = df.loc[keep_mask].copy()

    filtered_values = filtered_df[attribute].to_numpy()

    if clip_quantiles is not None:
        q_low, q_high = clip_quantiles
        lo = np.quantile(filtered_values, q_low)
        hi = np.quantile(filtered_values, q_high)
        filtered_values = np.clip(filtered_values, lo, hi)
    else:
        lo = filtered_values.min()
        hi = filtered_values.max()

    if hi - lo < 1e-12:
        filtered_df[attribute] = 0.0
    else:
        filtered_df[attribute] = (filtered_values - lo) / (hi - lo)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = _drop_helper_columns(filtered_df)
    result_df.to_csv(output_csv_path, index=False)

    if verbose:
        print(f"{attribute}: kept {len(result_df)} / {len(df)} rows")
        print(f"CSV сохранён в {output_csv_path}")

    return result_df


if __name__ == "__main__":

    fixed_ranges = {
        "pitch_range": (0.20, 0.75),
        "note_density": (0.20, 0.75),
        "contour": (-0.20, 0.20),
    }

    note_density_csv = Path("./vgmusic_midi_note_density_thinned.csv")
    contour_csv = Path("./vgmusic_midi_contour_thinned.csv")

    thin_tail_and_rescale(
        csv_path=Path("./vgmusic_midi.csv"),
        output_csv_path=note_density_csv,
        attribute="note_density",
        soft_threshold=0.75,
        hard_threshold=0.9,
        keep_mid_fraction=0.4,
        keep_high_fraction=0.2,
        random_state=42,
        clip_quantiles=(0.01, 0.99),
    )

    thin_contour_tail_and_rescale(
        csv_path=note_density_csv,
        output_csv_path=contour_csv,
        attribute="contour",
        soft_threshold=0.20,
        hard_threshold=0.35,
        keep_mid_fraction=0.4,
        keep_high_fraction=0.2,
        random_state=42,
        clip_abs_quantile=0.99,
    )

    save_midi_files_from_csv(
        csv_path=contour_csv,
        output_dir=Path("./ds/vgmusic_midi/thinned"),
        midi_dir=Path("./ds/vgmusic_midi/chunks_dedup"),
        copy_files=True,
    )

    '''
    undersample_by_attributes(
        midi_dir="./ds/vgmusic_midi/chunks_dedup",
        csv_path="./vgmusic_midi.csv",
        output_dir="./ds/vgmusic_midi/undersampled",
        bins_per_attr=8,
        max_per_stratum=6000,
        rarity_mode="per_attribute",
        binning_mode="fixed",
        fixed_bin_ranges=fixed_ranges,
        copy_files=True,
        random_state=42,
    )
    '''
    
    
    
    '''
    oversample_by_attributes(
        midi_dir="./ds/vgmusic_midi/undersampled",
        csv_path="./ds/vgmusic_midi/undersampled/attributes_undersampled.csv",
        output_dir="./ds/vgmusic_midi/oversampled",
        bins_per_attr=12,
        target_per_stratum=8100,
        rarity_mode="per_attribute",
        binning_mode="fixed",
        fixed_bin_ranges=fixed_ranges,
        max_abs_semitones=4,
        random_state=42,
    )
    '''
    
    '''
    augment_dataset(
        Path("./ds/vgmusic_midi/undersampled"),
        pitch_offsets=[-5, 5],
        velocity_offsets=[0, 0],
        duration_offsets=[0, 0],
        out_path=Path("./ds/vgmusic_midi/undersampled_aug"),
    )
    '''

    plot_attributes(contour_csv, 
                    attr_dict = {"path": [], 
                                "pitch_range": [],
                                "note_density": [],
                                "contour": []})
    
