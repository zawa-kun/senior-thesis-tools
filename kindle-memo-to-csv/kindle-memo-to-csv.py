"""Kindle メモの HTML からハイライト情報を CSV に書き出すスクリプト。"""
from pathlib import Path  # OSに依存しないパス操作ができる
from bs4 import BeautifulSoup
import csv


BASE_DIR = Path(__file__).resolve().parent# 親ディレクトリのパス
HTML_PATH = BASE_DIR / "notebook.htm"   # https://read.amazon.co.jp/notebook のHTMLに変換したHTMLファイルのパス
CSV_PATH = Path("../data/raw").resolve() / "highlights_jpn.csv"  # 出力先のCSVファイルの名前

records = []  # 収集したハイライト情報を格納するリスト


# Kindle ノート HTML を読み込み、パースして操作しやすいオブジェクトに変換する
with HTML_PATH.open("r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# id="kp-notebook-annotations"要素配下の各ノートブロックを抽出
blocks = soup.select("#kp-notebook-annotations div[id]")
print(len(blocks))
for block in blocks:
    """ハイライト部分の抽出"""
    # ブロック内からハイライトの要素を取得
    highlight_el = block.select_one(".kp-notebook-highlight")

    # ハイライト要素がなければ無視（メモのみのブロック等を排除）
    if not highlight_el:
        continue

    # ハイライトされているテキストを取得（半角スペースがある場合無くす）
    highlight = highlight_el.get_text(strip=True).replace(" ", "")

    """ロケーション部分の抽出"""
    location = ""
    # input フィールドに保存されている位置情報を優先して取得
    loc_input = block.find("input", id="kp-annotation-location")
    if loc_input and loc_input.has_attr("value"):
        location = loc_input["value"].strip()
    
    """メモ部分の抽出"""
    # メモが付いている場合だけテキストを取得する
    note_el = block.select_one(".kp-notebook-note")
    if note_el:
        note_words_str = note_el.get_text(" ", strip=True).replace("メモ", "") # 例 => 儒者 切支丹
        note_words_list = note_words_str.split() # 例 => ['儒者', '切支丹']
    else:
        note_words_list = []
        print(location, ":", "メモを書き忘れています！！！")
    

    # CSV 出力用に、位置・ハイライト・メモのセットを保存
    for note_word in note_words_list:
        records.append([location, highlight, note_word])

# 結果を CSV として書き出し、ヘッダ行を付与する
with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Location", "Highlight", "Note"])
    writer.writerows(records)

print(f"抽出完了：{len(records)} 件を {CSV_PATH.name} に保存しました。")
