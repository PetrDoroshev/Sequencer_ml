import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union

def check_attribute_correlation(
    csv_path: Union[str, Path],
    attr_columns: List[str],
    method: str = 'pearson',
    plot: bool = True,
    save_plot: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 6),
    cmap: str = 'coolwarm',
    annot: bool = True,
    fmt: str = '.2f'
) -> pd.DataFrame:
    """
    Загружает CSV с атрибутами и вычисляет корреляцию между указанными колонками.

    Параметры
    ----------
    csv_path : str или Path
        Путь к CSV-файлу.
    attr_columns : List[str]
        Список имён колонок, для которых нужно вычислить корреляцию.
    method : str, default='pearson'
        Метод корреляции: 'pearson', 'spearman', 'kendall'.
    plot : bool, default=True
        Построить тепловую карту.
    save_plot : str или Path, опционально
        Если указан, сохранить график в файл.
    figsize : tuple, default=(8,6)
        Размер фигуры.
    cmap : str, default='coolwarm'
        Цветовая карта для тепловой карты.
    annot : bool, default=True
        Показывать значения в ячейках.
    fmt : str, default='.2f'
        Формат чисел в аннотациях.

    Возвращает
    ----------
    pd.DataFrame
        Матрица корреляции.
    """
    # Загружаем данные
    df = pd.read_csv(csv_path)
    # Проверяем наличие колонок
    missing = [col for col in attr_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Колонки отсутствуют в CSV: {missing}")

    # Берём только нужные колонки
    data = df[attr_columns].copy()
    # Удаляем строки с пропусками (если есть)
    data = data.dropna()

    # Вычисляем корреляцию
    corr = data.corr(method=method)

    # Выводим числовую матрицу
    print(f"Матрица корреляции ({method}):")
    print(corr.to_string(float_format="%.3f"))
    print("\n")

    # Построение тепловой карты
    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f"Корреляция атрибутов ({method})")
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot, dpi=150)
        plt.show()

    return corr

if __name__ == "__main__":
    
    corr_pearson = check_attribute_correlation(
        csv_path="./vgmusic_midi_note_density_thinned.csv",
        attr_columns=['pitch_range', 'note_density', 'contour'],
        method='pearson',
        save_plot="correlation_pearson.png" 
    )