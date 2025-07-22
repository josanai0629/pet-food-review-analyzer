/**
 * OpenAI APIプロンプト設定ファイル
 * プロンプトを柔軟に変更できるように管理
 */

const PROMPTS = {
  // ラベル付け（10カテゴリ分類）プロンプト
  classification: `あなたはペットフードレビューの分類専門家です。
以下のレビューを、指定された10カテゴリのいずれか1つに分類してください。

カテゴリ：
1. 食べる - ペットが食べる、食いつきが良い、美味しそうに食べる
2. 食べない - ペットが食べない、食いつきが悪い、残す、拒否する
3. 吐く・便が悪くなる - 体調悪化、下痢、嘔吐、軟便
4. 吐き戻し・便の改善 - 体調改善、便の状態が良くなった、吐かなくなった
5. 値上がり/高い - 価格上昇、高価格への不満
6. 安い - 低価格、お得感、コスパが良い
7. 配送・梱包 - 配送状態、梱包への言及、届き方
8. 賞味期限 - 期限に関する言及、日付の問題
9. ジッパー - 保存用ジッパーへの言及、袋の開け閉め
10. その他 - 上記に該当しない内容

レビュー: "{review}"

回答は必ずカテゴリ名のみを返してください（例：食べる）。`,

  // センチメント分析プロンプト（sentimentAnalysisPrompt.txtの内容を使用）
  sentiment: `You are an AI specialized in advanced sentiment analysis. Analyze the sentiment of the provided Japanese text and return one of the following labels: "Positive," "Negative," or "Neutral." Additionally, provide a concise reason for your judgment (in Japanese) and a numerical score from -100 to 100 (-100 being most negative, 100 being most positive). Format your output as follows in JSON: {"label": "sentiment label", "score": numerical score, "reason": "reason for judgment in Japanese"}

**Important instructions for interpreting Japanese text:**
* **Deeply understand Japanese nuances and context.** Focus not only on the superficial meaning of words but also on how they are used in sentences and how the speaker's emotions are expressed.
* **Do not overlook simple affirmative and negative expressions in Japanese.** For instance, common Japanese phrases that indicate affirmation or positive assessment (conceptually, patterns like 'X is good,' or 'Y is wonderful') should be interpreted positively. Conversely, common Japanese phrases indicating negation or negative assessment (conceptually, patterns like 'X is not good,' or 'Y is disappointing') should be interpreted negatively.
* **As a specific example, if a Japanese text expresses a concept similar to 'grain-free is good' – where 'good' is a clear affirmative evaluation of the subject 'grain-free' – this should be judged as a POSITIVE sentiment.** When a text includes such a positive evaluation of a fact or feature, accurately capture that sentiment.
* **Pay close attention to adjectives, auxiliary verbs, and sentence-ending expressions in Japanese, as they often convey crucial emotional cues.**

Please also refer to the conceptual examples of word categories below that should generally be considered positive or negative when their Japanese equivalents appear in the text. Use these concepts as a reference for calculating the score, but **do not be limited by this list; judge comprehensively from the overall context.**

**Conceptual Positive Word Categories (look for Japanese equivalents):**
"eat/eating", "improved bowel movements", "cheap/inexpensive", "eager eating/good appetite", "looks delicious", "delicious", "satisfaction", "happiness", "relief/security", "desire to repurchase", "good quality", "positive result"

**Conceptual Negative Word Categories (look for Japanese equivalents):**
"delivery/packaging issues", "expensive/price increase", "vomiting/regurgitation", "worsened bowel movements/digestive issues", "expiration date concerns", "desire for packaging improvement (e.g., wishing for a zipper)", "refusal to eat", "disappointment", "dissatisfaction", "desire for improvement", "poor quality", "negative result"

Review: "{review}"`
};

// プロンプトのバージョン管理
const PROMPT_VERSION = "1.0.0";

// カテゴリ定義（参照用）
const CATEGORIES = [
  '食べる',
  '食べない',
  '吐く・便が悪くなる',
  '吐き戻し・便の改善',
  '値上がり/高い',
  '安い',
  '配送・梱包',
  '賞味期限',
  'ジッパー',
  'その他'
];

// CommonJS exports
module.exports = { PROMPTS, CATEGORIES, PROMPT_VERSION };