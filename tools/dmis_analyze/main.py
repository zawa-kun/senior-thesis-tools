"""
翻訳手法・異文化感受性自動分析システム

このプログラムは以下を行います：
1. CSVファイルから日英対訳データを読み込み
2. Google Gemini APIで翻訳手法と異文化感受性を分析
3. 結果を元のCSVに追加して保存
"""
import os
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm  # 進捗バー（オプション）
from pathlib import Path

# ========================================
# 設定・定数
# ========================================

# .envファイルから環境変数を読み込み
load_dotenv()

# --- Gemini APIの設定 ---
# APIKeyの設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Geminiのモデルの設定
# 無料かつ最新のモデルはリリースノートで確認
MODEL_NAME = "gemini-2.5-flash"

# レート制限対応：10 RPM = # 6秒に1リクエスト
# 無料範囲内で収まるように設定
REQUEST_INTERVAL = 6  # 秒

# リトライ設定
MAX_RETRIES = 3  # 最大リトライ回数
RETRY_DELAY = 5  # リトライ待機時間（秒）


# --- ファイルパスの設定 ---
INPUT_CSV_PATH = "alignment_ineko.csv"
# INPUT_CSV_PATH = "test_data.csv" # test
OUTPUT_CSV_PATH = "dmis_ineko.csv"
# OUTPUT_CSV_PATH = "test_data_analyzed.csv"
LOG_FILE = "error_log.txt"  # エラーログ


def create_prompt(highlight_jp, highlight_en, note="", annotation=""):
    """
    Gemini APIに送るプロンプトを生成

    Args:
        highlight_jp (str): 日本語原文
        highlight_en (str): 英訳
        note (str): メモ・キーワード
        annotation (str): 注釈
        
    Returns:
        str: 生成されたプロンプト
    """
    # 空欄の場合は「なし」に変換
    note = note if pd.notna(note) and note.strip() else "なし"
    annotation = annotation if pd.notna(annotation) and annotation.strip() else "なし"
    
    prompt = f"""以下の日本語原文と英訳を、一つの文化要素（キーワード）に焦点を当てて分析してください。

【日本語原文】
{highlight_jp}

【英訳】
{highlight_en}

【キーワード（文化的要素）】
{note}

【注釈】
{annotation}

---

あなたの役割：
翻訳者本人は DMIS の**統合段階**にいるという前提で、
翻訳処理そのものは文化項目ごとに DMIS の任意の段階（否認〜統合）を意図的に**“演じて”選択する**、
という研究モデルで分析してください。DMISの分類は、**読者に与えたい文化差の「見え方」**に基づきます。

分析手順（厳守）：

1. 翻訳手法を、ヴィネイ&ダルベルネの7分類の中から最も適切なものを1つだけ選択する。
【翻訳手法】
- 借用：原語をそのまま持ち込む（例：畳 → tatami）
- 仮借：構造を保った直訳（例：science fiction → 科学幻想）
- 直訳：文構造を基本維持した自然な逐語訳
- 転換：品詞・文法カテゴリを変えて意味保持（名詞→動詞など）
- 調整：視点／概念の枠組みを変える（抽象↔具体、部分↔全体）
- 等価：慣用表現をTL側の自然な表現へ置換
- 翻案：文化状況そのものを置き換えて意味機能を揃える


2. キーワード（文化的要素）の扱いに基づき、翻訳者が意図的に“どの DMIS モード（否認〜統合）を採用したか”を1つ選択する。

【DMISの翻訳適用（翻訳者が“演じる”文化モード）】
 - 否認：**文化差が存在しない視点**。文化的要素やその機能・情報が**完全に削除**され、痕跡を残さない。
 - 防衛：**他文化を歪める視点**。他文化を異質的・劣ったものとして誇張・敵対視させる（稀）。
 - 最小化：**共通点に収束する視点**。差異を薄め、普遍的な意味や共通概念に置き換える（例：等価）。
 - 受容：**差異をそのまま残す視点**。文化差を認めたうえで保持する。読者に意識的な理解を要求する（例：借用・直訳）。
 - 適応：**他文化視点に切り替える視点**。文化差を認識し、ターゲット読者が**機能的に理解できる**別の等価な要素に置き換える（例：機能的翻案）。
 - 統合：**両文化を融合する視点**。機能的等価性を超え、要素の**象徴的・比喩的な意味を再構築**し、両文化の概念を柔軟に操作する（例：概念の昇華を伴う翻案・調整）。

2.5. **【DMIS選択のチェックポイント（再現性向上のため）】**
 以下の基準で最終確認を行う。
 - **否認**：文化的要素/機能が、訳文で完全に削除され、痕跡を残していないか？
 - **最小化**：訳語が、両文化に共通する最も一般的な上位概念に収束しているか？（直訳・等価・調整で多用）
 - **受容**：文化的固有性が維持され、読者にその差異の認識を求めているか？（借用・直訳で多用）
 - **適応**：訳語が、ターゲット読者が違和感なく**機能的役割**を理解できる等価物に置き換えられているか？（翻案・調整で多用）
 - **統合**：訳語が、単なる機能的等価性を超えて、**象徴的・比喩的な概念を再構築**しているか？


3. 100文字以内で、翻訳手法と DMIS モードを選択した根拠を、
必ず原文と訳文の具体例を引用しながら簡潔に述べる。
※ 文全体ではなく、必ず「キーワード（文化的要素）」の翻訳を中心に扱うこと。キーワードの補足情報として、要素の周りで補足されている場合もあるので状況に応じて周りも交えて述べる事。

回答形式（必ずこの1行形式）：
**Noteの英訳での対応語/対応句,翻訳手法,DMIS段階,備考**

例：
the sensei,借用,受容,原文「先生」を「the sensei」と借用し、日本語固有の敬称の文化的差異をそのまま提示し、読者にその受容を求めている。

※ 必ず4項目をカンマ区切りで1行のみ出力する。
"""
    return prompt


# ========================================
# Gemini API呼び出し
# ========================================

def call_gemini_api(prompt, retries=MAX_RETRIES):
    """
    Gemini APIを呼び出して分析結果を取得
    
    Args:
        prompt (str): 送信するプロンプト
        retries (int): リトライ回数
        
    Returns:
        str: APIからの回答（失敗時はNone）
    """
    for attempt in range(retries):
        try:
            # Geminiモデルを取得
            model = genai.GenerativeModel(MODEL_NAME)
            
            # プロンプトを送信
            response = model.generate_content(prompt)
            
            # 回答テキストを取得
            if response and response.text:
                return response.text.strip()
            else:
                log_error(f"API応答が空です（試行 {attempt + 1}/{retries}）")
                
        except Exception as e:
            log_error(f"API呼び出しエラー（試行 {attempt + 1}/{retries}）: {str(e)}")
            
            # 最後の試行でなければ待機してリトライ
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    
    # すべてのリトライが失敗
    return None


# ========================================
# 回答パース
# ========================================

def parse_response(response_text):
    """
    APIの回答をパースして4項目に分割
    
    Args:
        response_text (str): APIからの回答
        
    Returns:
        tuple: (翻訳手法, 異文化感受性, 備考)
    """
    if not response_text:
        return "API呼び出し失敗", "API呼び出し失敗", "APIからの応答がありませんでした"
    
    try:
        # カンマで分割（最大4分割）
        parts = response_text.split(',', 3)

        # 4項目が揃っているか確認
        if len(parts) >= 4:
            translated_term = parts[0].strip()
            method = parts[1].strip()
            sensitivity = parts[2].strip()
            note = parts[3].strip()
            return translated_term, method, sensitivity, note
        else:
            # 形式が不正な場合
            return "解析エラー", "解析エラー", f"形式不正: {response_text}"
            
    except Exception as e:
        # エラー発生時（4項目を返すように修正)
        return "解析エラー", "解析エラー", f"パースエラー: {str(e)}"


# ========================================
# ログ出力
# ========================================

def log_error(message):
    """
    エラーログをファイルに出力
    
    Args:
        message (str): ログメッセージ
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    
    # コンソールにも表示
    print(f"⚠️  {message}")
    
    # ファイルに追記
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message)


# ========================================
# CSV読み込み
# ========================================

def load_csv(file_path):
    """
    CSVファイルを読み込み
    
    Args:
        file_path (str): CSVファイルのパス
        
    Returns:
        DataFrame: 読み込んだデータ（失敗時はNone）
    """
    try:
        # UTF-8で読み込み
        df = pd.read_csv(file_path, encoding="utf-8")
        
        # 必須列の確認
        required_columns = ["Highlight_JP", "Highlight_EN"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ エラー: 必須列が見つかりません: {missing_columns}")
            return None
        
        print(f"✅ CSVファイル読み込み成功: {len(df)}行")
        print(f"📋 列名: {df.columns.tolist()}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ エラー: ファイルが見つかりません: {file_path}")
        print(f"💡 ヒント: {file_path} をこのプログラムと同じフォルダに置いてください")
        return None
        
    except Exception as e:
        print(f"❌ CSVファイル読み込みエラー: {str(e)}")
        return None


# ========================================
# メイン処理
# ========================================

def main():
    """
    メイン処理フロー
    """
    print("=" * 60)
    print("🚀 翻訳手法・異文化感受性自動分析システム 起動")
    print("=" * 60)
    print()
    
    # ========================================
    # 1. APIキーの確認
    # ========================================
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
        print("❌ エラー: APIキーが設定されていません")
        print()
        print("📝 設定方法：")
        print("1. .env ファイルを作成")
        print("2. 以下の内容を記述：")
        print("   GEMINI_API_KEY=あなたのAPIキー")
        print()
        print("APIキー取得: https://aistudio.google.com/app/apikey")
        return
    
    # Gemini APIを初期化
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API初期化完了")
    print()
    
    # ========================================
    # 2. CSVファイルの読み込み
    # ========================================
    df = load_csv(INPUT_CSV_PATH)
    if df is None:
        return
    
    print()
    
    # 出力列が存在しない場合は追加
    if "文化的要素の英訳句/語" not in df.columns:
        df["文化的要素の英訳句/語"] = ""
    if "ヴィネイとダルベルネの翻訳7分類" not in df.columns:
        df["ヴィネイとダルベルネの翻訳7分類"] = ""
    if "ベネットの異文化感受性モデル" not in df.columns:
        df["ベネットの異文化感受性モデル"] = ""
    if "備考" not in df.columns:
        df["備考"] = ""
    
    # ========================================
    # 3. 行ごとのループ処理
    # ========================================
    total_rows = len(df)
    success_count = 0
    error_count = 0
    
    print(f"📊 処理開始: {total_rows}行を処理します")
    print(f"⏱️  推定時間: 約{total_rows * REQUEST_INTERVAL // 60}分")
    print()
    
    # 進捗バー付きループ
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="処理中"):
        # データ取得
        highlight_jp = row.get("Highlight_JP", "")
        highlight_en = row.get("Highlight_EN", "")
        note = row.get("Note", "")
        annotation = row.get("注釈", "")
        
        # 必須項目チェック
        if pd.isna(highlight_jp) or pd.isna(highlight_en):
            log_error(f"行{index + 2}: 必須データが欠落")
            df.at[index, "ヴィネイとダルベルネの翻訳7分類"] = "データ欠落"
            df.at[index, "ベネットの異文化感受性モデル"] = "データ欠落"
            df.at[index, "備考"] = "Highlight_JPまたはHighlight_ENが空です"
            error_count += 1
            continue
        
        # プロンプト生成
        prompt = create_prompt(highlight_jp, highlight_en, note, annotation)
        
        # API呼び出し
        response = call_gemini_api(prompt)
        
        # 回答をパース
        translated_term, method, sensitivity, remark = parse_response(response)
        
        # DataFrameに書き込み
        df.at[index, "文化的要素の英訳句/語"] = translated_term
        df.at[index, "ヴィネイとダルベルネの翻訳7分類"] = method
        df.at[index, "ベネットの異文化感受性モデル"] = sensitivity
        df.at[index, "備考"] = remark
        
        # 成功カウント
        if "エラー" not in method and "失敗" not in method:
            success_count += 1
        else:
            error_count += 1
        
        # レート制限対応：次のリクエストまで待機
        time.sleep(REQUEST_INTERVAL)
    
    # ========================================
    # 4. 結果をCSVに保存
    # ========================================
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
        print()
        print("=" * 60)
        print("✅ 処理完了！")
        print("=" * 60)
        print(f"📁 保存先: {OUTPUT_CSV_PATH}")
        print(f"✅ 成功: {success_count}行")
        print(f"❌ エラー: {error_count}行")
        print()
        
        if error_count > 0:
            print(f"⚠️  エラー詳細はログファイルを確認: {LOG_FILE}")
        
    except Exception as e:
        print(f"❌ CSVファイル保存エラー: {str(e)}")


# ========================================
# プログラム起動
# ========================================

if __name__ == "__main__":
    main()