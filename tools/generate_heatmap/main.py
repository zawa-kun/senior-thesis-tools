import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 日本語フォントに設定---
plt.rcParams['font.family'] = 'MS Gothic'

# --- CSV 読み込み ---
df = pd.read_csv("test_data.csv",encoding="utf-8", engine="python")

# --- 必要な2列だけ抽出 ---
col_method = "翻訳技法"
col_dmis   = "DMIS"

# --- 前処理：文字列化前後スペース削除、欠損は空文字に ---
df[col_method] = df[col_method].astype(str).str.strip().fillna("")
df[col_dmis]   = df[col_dmis].astype(str).str.strip().fillna("")

# 想定するカテゴリ
dmis_order = ['Denial', 'Minimization', 'Acceptance', 'Adaptation']
methods_order = ['Amplification', 'Borrowing', 'Established equivalent', 'Reduction', 'Description', 'Generalization', 'Adaptation', 'Literal translation', 'Particulization', 'Adaptation']

# --- 4. 未知ラベルの存在チェック（ログ出力） ---
unknown_dmis = sorted(set(df[col_dmis].unique()) - set(dmis_order))
unknown_methods = sorted(set(df[col_method].unique()) - set(methods_order))
if unknown_dmis:
    print("未定義の DMIS ラベル（修正が必要）:", unknown_dmis)
if unknown_methods:
    print("未定義の翻訳手法ラベル（修正が必要）:", unknown_methods)

# --- 3. クロス集計（頻度表） ---
crosstab = pd.crosstab(df[col_method], df[col_dmis])
crosstab = crosstab.reindex(index=methods_order, columns=dmis_order, fill_value=0)



# --- 8. ヒートマップ描画 ---
plt.figure(figsize=(10, 7))
sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'count'})
plt.title("DMIS x translation_method -- Frequency Heatmap", fontsize=14)
plt.xlabel("DMIS")
plt.ylabel("translation method")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("dmis_translation_heatmap.png", dpi=300)  # 保存
plt.show()
