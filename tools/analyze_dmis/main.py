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
INPUT_CSV_PATH = "test.csv"
OUTPUT_CSV_PATH = "test_analyzed.csv"
LOG_FILE = "error_log.txt"  # エラーログ


def create_prompt(highlight_jp, highlight_en, note="", annotation="",translation_correspondence=""):
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
    translation_correspondence = translation_correspondence if pd.notna(translation_correspondence) and translation_correspondence.strip() else "なし"

    prompt = f"""以下の日本語原文と英訳を、一つの文化要素（キーワード）に焦点を当てて分析してください。

【日本語原文】
{highlight_jp}

【英訳】
{highlight_en}

【キーワード（文化的要素）】
{note}

【注釈】
{annotation}

【文化的要素の対応訳】
{translation_correspondence}

---

あなたの役割：
異文化コミュニケーションおよび翻訳研究の専門家として、
「日本に関する知識が皆無の読者（Aさん）」が、翻訳テキストを通じて文化的要素をどの程度の深さで理解できる状態にあるかを、
客観的な第三者（統合の視点を持つ分析者）として判定してください。

分析手順（厳守）：

1. 文脈の確認：
   入力された【文化的要素の対応訳】だけでなく、必ず【英訳】全体の文脈を確認してください。
   特に、対応訳がどのように形容されているか（形容詞）、どう扱われているか（動詞）に注目してください。

2. 段階の判定：
   以下の【DMIS定義表】に基づき、翻訳テキストがAさんを強制的に置く認知段階を1つだけ特定してください。

【DMIS定義表（6段階モデル）】

| DMIS段階 | 判定基準（翻訳テキストの特性） | 読者（Aさん）の文化的要素に対する認知状態 |
| :--- | :--- | :--- |
| **Denial (否認)** | **【削除・不可視化】**<br>文化的要素が完全に削除・省略されている。または全く意味不明な訳語。 | **認識不可 / 無関心**<br>要素をノイズとして無視するしかない状態。 |
| **Defense (防衛)** | **【異化の強調・不全】**<br>不自然な直訳や過度な異国趣味により、「奇妙なもの」「不気味なもの」として提示されている。<br>（否定的な形容詞や文脈が含まれる場合もここ） | **拒絶 / 脅威**<br>「奇妙だ」「劣っている」と否定的に捉える状態。 |
| **Minimization (最小化)** | **【自文化への置き換え】**<br>自文化の既知の概念（等価物）に完全に置換している。<br>★重要：文化的要素の**固有性（違い）が消えている**場合はここ。 | **同化 / 普遍的理解**<br>「私の国と同じだ」と誤って処理し、固有性を意識しない状態。 |
| **Acceptance (受容)** | **【差異の提示】**<br>音写などでそのまま提示し、安易な置き換えを避けている。補足説明はほぼない。<br>★重要：**固有性（違い）が残っている**場合はここ。 | **差異の認識**<br>「私の国とは違う」と認識し、受け入れようとする状態。 |
| **Adaptation (適応)** | **【橋渡し・厚い記述】**<br>詳細な補足や背景説明（文中説明や訳注）を加え、機能や文脈を論理的・感情的に説明している。<br>（音写＋補足説明など） | **共感 / 文脈的理解**<br>「向こうではこう機能する」と一時的に視点を転換できる状態。 |
| **Integration (統合)** | **【比較・相対化の提示】**<br>自文化と異文化の価値観を対比させ、読者自身の常識やアイデンティティを問い直す記述がある。 | **相対化・再構築**<br>「私の常識も一つの偏りだ」とメタ的に自己内省する状態。 |

回答作成のルール：
- **文脈依存性：** 対応訳単体ではなく、文全体での扱われ方を根拠にすること。
- **最小化 vs 適応：** 固有性が消えていれば「最小化」、残っていれば「受容」以上とする。
- **適応 vs 統合：** 読者の価値観への問いかけがなければ「適応」に留めること。

回答形式（必ずこの1行形式）：
**DMIS段階,そのDMISであると考えられる理由**

出力例：
Denial,原文にあった色彩情報が訳文では完全に削除されており、読者はその要素の存在自体を認知できないため。
Defense,文脈なしに直訳しており、読者には意味不明な異物として奇妙に映り、理解を拒絶させる可能性があるため。
Minimization,「書生」を一般的な「student」に置き換えており、読者は固有性を意識せず自国の学生と同じものとして処理するため。
Acceptance,補足なく音写しており、読者に意味は不明確ながらも「自文化にはない固有の概念」として差異を認識させているため。
Adaptation,音写に加え「布団で覆われた暖房器具」という機能説明があり、読者はその形状と用途を具体的にイメージし共感できるため。
Integration,西洋の美意識と対比して説明しており、読者自身の「美」に対する固定観念を相対化させ、複合的な視点を与えているため。

※ 必ず2項目をカンマ区切りで1行のみ出力する。
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
    APIの回答をパースして2項目に分割
    
    Args:
        response_text (str): APIからの回答
        
    Returns:
        tuple: (DMIS, DMISの選出理由)
    """
    if not response_text:
        return "API呼び出し失敗", "API呼び出し失敗", "APIからの応答がありませんでした"
    
    try:
        # カンマで分割（最大2分割）
        parts = response_text.split(',', 1)

        # 3項目が揃っているか確認
        if len(parts) >= 2:
            dmis = parts[0].strip()
            reason = parts[1].strip()
            return dmis, reason
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
    if "DMIS" not in df.columns:
        df["DMIS"] = ""
    if "DMISの選出理由" not in df.columns:
        df["DMISの選出理由"] = ""
    
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
        translation_correspondence = row.get("文化的要素の対応訳","")

        # 必須項目チェック
        if pd.isna(highlight_jp) or pd.isna(highlight_en):
            log_error(f"行{index + 2}: 必須データが欠落")
            df.at[index, "DMIS"] = "データ欠落"
            df.at[index, "DMISの選出理由"] = "Highlight_JPまたはHighlight_ENが空です"
            error_count += 1
            continue
        
        # プロンプト生成
        prompt = create_prompt(highlight_jp, highlight_en, note, annotation, translation_correspondence)
        
        # API呼び出し
        response = call_gemini_api(prompt)
        
        # 回答をパース
        dmis, reason = parse_response(response)
        
        # DataFrameに書き込み
        df.at[index, "DMIS"] = dmis
        df.at[index, "DMISの選出理由"] = reason
        
        # 成功カウント
        if "エラー" not in dmis and "失敗" not in dmis:
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