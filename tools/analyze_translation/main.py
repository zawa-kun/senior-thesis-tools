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
INPUT_CSV_PATH = "proc_alignment_ineko.csv"
OUTPUT_CSV_PATH = "raw_analyzed_ineko.csv"
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
翻訳研究者として、指定されたキーワード（文化的要素）がどのように翻訳されているかを分析し、
以下の翻訳技法分類に基づいて分類してください。

分析手順（厳守）：

1. 翻訳技法を、以下の中から最も適切なものを1つだけ選択する。
【翻訳技法】
- Borrowing: 日本語の語句をそのまま音写（ローマ字化）して使用する。`英訳`にも`注釈`にも一切補足説明がない場合のみ適用する。
    - 例：「先生」-> "Sensei"

- Amplification: 「Borrowing（借用）」に加え、原文にない詳細（情報や説明的パラフレーズ）を加える。`注釈`に補足説明がある場合もAmplificationにあたる。

    - 例：「先生」-> "Sensei,〈詳細〉"、「先生」 -> "Sensei"+注釈でSenseiについての説明、「先生」など
    - 補足：この「詳細」には、訳文の脚注や注釈（{annotation}）に含まれる説明も含むこととする。

- Calque: 外国語の語句を逐語訳して取り入れる。語順は保たれる必要はない。

    - 例：「右大臣」-> "Minister of the Right" 語義的な Calque, 「森林浴」-> "forest bathing" 構造的な Calque

- Literal translation: 語句を逐語的に訳すが、形式・機能・意味が一致する場合に限る。辞書的な意味通り。

    - 例：「父」->"father"

- Established equivalent: 辞書や慣用表現として認められている等価語を使用する

    - 例：「酒」 -> "rice wine"、「畳」-> "Straw mat"

- Generalization: より一般的・中立的な用語を使用する。抽象化すること。

    - 例：「書生」 -> "student"

- Particularization: より具体的・精密な用語を使用する(文脈から具体名を特定する場合等)

    - 例：「花」-> "cherry blossoms"

- Description: 原文の言葉を使わずに用語をその形態や機能の説明に置き換える
    - 例：「こたつ」 -> "a heated table covered with a quilt"）

- Adaptation: 原文の文化的要素を、ターゲット文化の要素に置き換える

    - 例：「サッカー」->"baseball"、「将棋」-> "chess"

- Modulation: 視点や認知的カテゴリーを変更する。肯定・否定の反転や、受動・能動の切り替え、部分と全体の関係変更。

    - 例：「死ぬ」-> "stop living", 「父になる」-> "have a child"）

- Reduction: 原文の情報項目を省略する。原文にあった文化的要素が訳文に無くなる。

    - 例：「鳶色のカステラ」-> "cake"(「鳶色」という情報が消えている)

2. 100文字以内で、翻訳技法を選択した根拠を、必ず原文と訳文の具体例を引用しながら簡潔に述べる。
※ 文全体ではなく、必ず「キーワード（文化的要素）」の翻訳を中心に扱うこと。

回答形式（必ずこの1行形式）：
**文化的要素の対応訳,翻訳技法,翻訳技法の選出理由**

例：
Sensei,Borrowing,原文「先生」を「Sensei」と音写し、補足説明を加えていないため。		
sake, a kind of Japanese rice wine,Amplification,原文「酒」を「sake」と借用しつつ「a kind of Japanese rice wine」と説明（増幅）を加えているため。		
Minister of the Right,Calque,「右大臣」を「Minister of the Right」と語義的に逐語訳（仮借）しているため。		
father,Literal translation,「父」を「father」と辞書的な意味の通りに直訳し、形式・機能・意味が一致しているため。		
Straw mat,Established equivalent,「畳」を英語圏で定着している訳語「Straw mat」で対応させているため。		
student,Generalization,「書生」という特定の身分の学生を、より一般的な用語である「student」に一般化しているため。		
cherry blossoms,Particularization,「花」を、文脈上最も具体的な種である「cherry blossoms」に具体化しているため。		
a heated table covered with a quilt,Description,「こたつ」という用語を使わず、その形態や機能（heated table covered with a quilt）を説明しているため。		
chess,Adaptation,「将棋」という文化要素を、ターゲット文化圏で機能的に近い「chess」に置き換えているため。		
stop living,Modulation,「死ぬ」という概念を、視点を変えて「stop living」という否定表現で訳しているため。		
cake,Reduction,原文にあった「鳶色」に相当する色情報が訳文の「cake」では完全に省略されているため。


※ 必ず3項目をカンマ区切りで1行のみ出力する。
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
    APIの回答をパースして3項目に分割
    
    Args:
        response_text (str): APIからの回答
        
    Returns:
        tuple: (文化的要素の対応訳, 翻訳技法, 翻訳技法の選出理由)
    """
    if not response_text:
        return "API呼び出し失敗", "API呼び出し失敗", "APIからの応答がありませんでした"
    
    try:
        # カンマで分割（最大3分割）
        parts = response_text.split(',', 2)

        # 3項目が揃っているか確認
        if len(parts) >= 3:
            translated_term = parts[0].strip()
            method = parts[1].strip()
            reason = parts[2].strip()
            return translated_term, method, reason
        else:
            # 形式が不正な場合
            return "解析エラー", "解析エラー", f"形式不正: {response_text}"
            
    except Exception as e:
        # エラー発生時
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
    if "文化的要素の対応訳" not in df.columns:
        df["文化的要素の対応訳"] = ""
    if "翻訳技法" not in df.columns:
        df["翻訳技法"] = ""
    if "翻訳技法の選出理由" not in df.columns:
        df["翻訳技法の選出理由"] = ""
    
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
            df.at[index, "翻訳技法"] = "データ欠落"
            df.at[index, "翻訳技法の選出理由"] = "Highlight_JPまたはHighlight_ENが空です"
            error_count += 1
            continue
        
        # プロンプト生成
        prompt = create_prompt(highlight_jp, highlight_en, note, annotation)
        
        # API呼び出し
        response = call_gemini_api(prompt)
        
        # 回答をパース
        translated_term, method, reason = parse_response(response)
        
        # DataFrameに書き込み
        df.at[index, "文化的要素の対応訳"] = translated_term
        df.at[index, "翻訳技法"] = method
        df.at[index, "翻訳技法の選出理由"] = reason
        
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