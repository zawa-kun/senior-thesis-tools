from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- パスの設定 ---
# 英語版のテキストファイルのパス
TXT_PATH = Path(__file__).parent.parent / "data" / "raw" / "ineko_ver.txt"
# 日本語のCSVのパス
CSV_PATH = Path(__file__).parent.parent / "data" / "processed" / "highlights_clean.csv"
# アラインメントのパス
ALIGNMENT_PATH = Path(__file__).parent.parent / "data" / "raw" / "alignment.csv"


"""文単位の分解"""
# --- テキストファイルの読み込み ---
with TXT_PATH.open("r", encoding="utf-8") as f:
    text_en = f.read()

# --- 改行やノイズの削除 --- 
text = text_en.replace("\n", " ").replace("\r", " ")

# --- 本編開始位置を検出 ---
start_idx = text.find("PART I")
text_main = text[start_idx:]

# --- 文分割 ---
sentences = sent_tokenize(text_main)

"""CSVの読み込み"""
# --- csv読み込み ---
df_jp = pd.read_csv(str(CSV_PATH))

"""意味類似度で自動マッチング"""
# モデル選択
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

emb_jp = model.encode(df_jp["Highlight"].tolist(), convert_to_tensor=True)
emb_en = model.encode(sentences, convert_to_tensor=True)

matches = []
for i, e in enumerate(emb_jp):
    sims = util.cos_sim(e, emb_en)[0]
    best_idx = sims.argmax().item()
    matches.append({
        "Location": df_jp.loc[i, "Location"],
        "Highlight_JP": df_jp.loc[i, "Highlight"],
        "Note": df_jp.loc[i, "Note"],
        "Highlight_EN": sentences[best_idx],
        "Similarity": sims[best_idx].item()
    })


df_out = pd.DataFrame(matches)
df_out.to_csv(ALIGNMENT_PATH, index=False)