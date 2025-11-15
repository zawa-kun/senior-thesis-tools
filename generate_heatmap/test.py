import pandas as pd
import matplotlib.pyplot as plt

# --- 日本語フォントに設定---
plt.rcParams['font.family'] = 'MS Gothic'

# --- CSV 読み込み ---
df = pd.read_csv("dmis_ineko.csv",encoding="utf-8", engine="python")
print("dmis:",df["dims_stage"].unique())
print("翻訳手法:",df["translation_method"].unique())