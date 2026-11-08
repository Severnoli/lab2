"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    """Корреляция Пирсона каждого признака с MRI_Count (пропуски отбрасываются попарно)."""
    return {f: float(group[f].corr(group[TARGET], method="pearson")) for f in FEATURES}


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]

    women_corr = _mri_correlations(women)
    men_corr = _mri_correlations(men)

    # максимум модуля корреляции среди обеих групп
    best_feature, best_abs = None, -1.0
    for corr in (women_corr, men_corr):
        for feature, value in corr.items():
            if abs(value) > best_abs:
                best_feature, best_abs = feature, abs(value)

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=str(best_feature),
    )
