
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm   # опционально, для прогресс-бара (pip install tqdm)


def parse_args():
    parser = argparse.ArgumentParser(description="Случайная подвыборка MIDI-файлов dMelodies")
    parser.add_argument("source_dir", type=str, help="Путь к исходной папке с MIDI-файлами (raw_data)")
    parser.add_argument("output_dir", type=str, help="Путь к папке для сохранения выбранных файлов")
    parser.add_argument("num_samples", type=int, help="Количество файлов для выборки")
    parser.add_argument("--seed", type=int, default=42, help="Сид для воспроизводимости (по умолчанию 42)")
    parser.add_argument("--no-progress", action="store_true", help="Отключить прогресс-бар")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    num_samples = args.num_samplescl
    seed = args.seed
    use_progress = not args.no_progress

    if not source.is_dir():
        print(f"Ошибка: исходная папка {source} не существует.")
        return

    # Собираем все MIDI-файлы рекурсивно
    print(f"Сканирование {source}...")
    all_files = list(source.rglob("*.mid"))
    print(f"Найдено MIDI-файлов: {len(all_files)}")

    if len(all_files) == 0:
        print("Нет MIDI-файлов для выборки.")
        return

    if num_samples > len(all_files):
        print(f"Предупреждение: запрошено {num_samples}, но доступно только {len(all_files)}. Будет взято {len(all_files)}.")
        num_samples = len(all_files)

    # Перемешиваем с фиксированным сидом
    random.seed(seed)
    sampled_files = random.sample(all_files, num_samples)

    # Создаём выходную папку
    output.mkdir(parents=True, exist_ok=True)

    print(f"Копирование {num_samples} файлов в {output}...")
    iterator = tqdm(sampled_files, disable=not use_progress, desc="Копирование")
    for src in iterator:
        dst = output / src.name
        # Если имя конфликтует, можно добавить префикс или сохранять структуру,
        # но для простоты просто предупредим.
        if dst.exists():
            print(f"Предупреждение: файл {dst.name} уже существует, пропускаем.")
            continue
        shutil.copy2(str(src), str(dst))

    print("Готово!")


if __name__ == "__main__":
    main()