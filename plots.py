"""Графики для отчёта по ЛР №2. Запуск: python plots.py"""

import matplotlib

import itertools

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

sns.set_theme(style="whitegrid")

result = analyze_titanic(TitanicInput("titanic.csv"))
print(result)

titanic = pd.read_csv("titanic.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# пропуски по столбцам (только там, где они есть)
miss = pd.Series(result.missing_by_column)
miss = miss[miss > 0]
miss.plot(kind="bar", ax=axes[0], color="tab:red")
axes[0].set_title("Пропуски по столбцам")
axes[0].set_ylabel("Количество")

# средний возраст по классам
pd.Series(result.mean_age_by_pclass).plot(kind="bar", ax=axes[1], color="tab:blue")
axes[1].set_title("Средний возраст по Pclass")
axes[1].set_xlabel("Pclass")
axes[1].set_ylabel("Возраст")

# доля выживших по классам
pd.Series(result.survival_rate_by_pclass).plot(kind="bar", ax=axes[2], color="tab:green")
axes[2].set_title("Доля выживших по Pclass")
axes[2].set_xlabel("Pclass")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("plot1.png", dpi=120)
plt.show()

brain_result = analyze_brain_correlations(BrainDataInput("brainsize.txt"))
print(brain_result)

brain = pd.read_csv("brainsize.txt", sep="\t", na_values="NA")
features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
men = brain[brain["Gender"] == "Male"]
women = brain[brain["Gender"] == "Female"]

corr_table = pd.DataFrame({
    "Женщины": brain_result.women_mri_correlation,
    "Мужчины": brain_result.men_mri_correlation,
})
print(corr_table)

fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharey="row")
for row, (name, group) in enumerate([("Женщины", women), ("Мужчины", men)]):
    for col, feature in enumerate(features):
        ax = axes[row, col]
        sns.regplot(data=group, x=feature, y="MRI_Count", ax=ax, ci=None)
        ax.set_title(f"{name}: {feature}")
plt.tight_layout()
plt.savefig("plot2.png", dpi=120)
plt.show()

cols = features + ["MRI_Count"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, group) in zip(axes, [("Женщины", women), ("Мужчины", men)]):
    sns.heatmap(group[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Корреляции Пирсона: {name}")
plt.tight_layout()
plt.savefig("plot3.png", dpi=120)
plt.show()
