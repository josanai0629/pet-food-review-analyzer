/**
 * データ処理API - バッチ処理対応
 * Vercel Functions用エンドポイント（OpenAI API通信隠蔽）
 */

const { PROMPTS, CATEGORIES } = require('./prompts.js');

module.exports = async function handler(req, res) {
  // CORS設定
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // プリフライトリクエスト処理
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // 🔐 デバッグモード判定
  const debugMode = req.headers['x-debug-mode'] === 'true';
  
  let review, type = 'classification', batchReviews, categories, batchId;
  
  if (debugMode) {
    // デバッグモード: 平文で受信
    ({ review, type = 'classification', batchReviews, categories, batchId } = req.body);
    console.log('🔍 [DEBUG] Plain text request received');
  } else {
    // 本番モード: 暗号化データを復号化
    try {
      const { p, c, t, id } = req.body;
      
      if (p) batchReviews = JSON.parse(Buffer.from(p, 'base64').toString('utf-8'));
      if (c) categories = JSON.parse(Buffer.from(c, 'base64').toString('utf-8'));
      type = t || 'classification';
      batchId = id;
      
      console.log('🔓 Decrypted request successfully');
    } catch (decryptError) {
      console.error('❌ Decryption failed, falling back to plain text:', decryptError);
      ({ review, type = 'classification', batchReviews, categories, batchId } = req.body);
    }
  }
  
  // 入力検証
  if (!review && !batchReviews) {
    return res.status(400).json({ error: 'Review text or batch reviews required' });
  }
  
  console.log(`🎯 Processing request - Type: ${type}, Batch: ${batchReviews ? batchReviews.length : 0} items, ID: ${batchId || 'single'}, Debug: ${debugMode}`);

  // APIキーの確認（デバッグ用ログ追加）
  console.log('Environment variables check:', {
    hasApiKey: !!process.env.OPENAI_API_KEY,
    keyPrefix: process.env.OPENAI_API_KEY ? process.env.OPENAI_API_KEY.substring(0, 7) + '...' : 'NOT_FOUND',
    nodeEnv: process.env.NODE_ENV
  });
  
  if (!process.env.OPENAI_API_KEY) {
    console.error('OPENAI_API_KEY not found in environment variables');
    console.error('Available env vars:', Object.keys(process.env));
    return res.status(500).json({ error: 'OpenAI API key not configured' });
  }

  try {
    // バッチ処理の場合
    if (batchReviews && Array.isArray(batchReviews)) {
      console.log(`🔄 Batch processing: ${batchReviews.length} reviews (${type})`);
      const results = [];
      
      for (let i = 0; i < batchReviews.length; i++) {
        const batchReview = batchReviews[i];
        try {
          const result = await processOpenAIRequest(batchReview.text, type, categories);
          
          // レスポンス形式統一
          if (type === 'classification') {
            results.push({
              id: batchReview.id || i,
              category: result.category,
              confidence: result.confidence,
              model: result.model
            });
          } else if (type === 'sentiment') {
            results.push({
              id: batchReview.id || i,
              sentiment: result.sentiment,
              score: result.score,
              reason: result.reason,
              model: result.model
            });
          }
          
          // レート制限対策：リクエスト間に短い待機
          if (i < batchReviews.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 50));
          }
        } catch (error) {
          console.error(`Batch processing error for review ${i}:`, error);
          results.push({
            id: batchReview.id || i,
            error: 'Processing failed',
            category: type === 'classification' ? 'エラー' : undefined,
            sentiment: type === 'sentiment' ? 'エラー' : undefined,
            score: type === 'sentiment' ? 0 : undefined,
            reason: type === 'sentiment' ? 'API処理失敗' : undefined
          });
        }
      }
      
      // 🔐 レスポンス暗号化
      if (debugMode) {
        return res.status(200).json({ results });
      } else {
        const encryptedResults = Buffer.from(JSON.stringify(results)).toString('base64');
        return res.status(200).json({ 
          d: encryptedResults,
          ts: Date.now(),
          v: "1.0",
          s: "ok"
        });
      }
    }

    // 単一レビューの処理
    const result = await processOpenAIRequest(review, type, categories);
    
    if (debugMode) {
      return res.status(200).json(result);
    } else {
      const encryptedResult = Buffer.from(JSON.stringify(result)).toString('base64');
      return res.status(200).json({
        d: encryptedResult,
        ts: Date.now(),
        v: "1.0",
        s: "ok"
      });
    }

  } catch (error) {
    console.error('OpenAI API error:', error);
    return res.status(500).json({ 
      error: 'Classification failed',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
}

/**
 * モデル設定
 * - classification: ファインチューニングモデル（分類専用最適化）
 * - sentiment: 汎用モデル（センチメント分析に適している）
 */
const MODEL_CONFIG = {
  classification: 'ft:gpt-4.1-nano-2025-04-14:personal::BwlMfAWg',
  sentiment: 'gpt-4o-mini'
};

/**
 * OpenAI APIリクエスト処理（カスタムカテゴリ対応・固定テンプレート）
 */
async function processOpenAIRequest(reviewText, type, customCategories = null) {
  let prompt;
  
  if (type === 'classification') {
    if (customCategories && customCategories.length > 0) {
      // 固定プロンプトテンプレートにカスタムカテゴリを埋め込み
      const categoryList = customCategories.map((cat, index) => `${index + 1}. ${cat}`).join('\n');
      
      // 🎯 固定プロンプトテンプレート（ReviewAnalysisPrompt.txtベース）
      prompt = `Act like a professional Japanese language annotator and sentiment analysis specialist. Your expertise lies in classifying and labeling Japanese customer reviews using a predefined taxonomy of labels.

Objective: Your task is to read a customer review written in Japanese and assign the most appropriate predefined Japanese label that best describes the content and sentiment of the review. These labels may refer to aspects such as product quality, service experience, price satisfaction, emotional tone, or specific features. Assume that the set of predefined labels is comprehensive and mutually exclusive.

Step-by-step instructions:

Step 1: Read the Japanese customer review provided between triple quotation marks.
Step 2: Analyze the review for thematic content, sentiment, and subject matter, considering any implicit or explicit expressions that may suggest specific labels.
Step 3: Select the single most appropriate label from the predefined set. If the review includes multiple themes, apply only the label that represents the primary focus or main concern expressed by the customer.

Predefined Labels:
${categoryList}

Customer Review:
"""${reviewText}"""

Output: Return ONLY the exact label name in Japanese (example: ${customCategories[0]}). Do not include explanations or additional text.`;
    } else {
      // デフォルトプロンプトを使用
      prompt = PROMPTS[type]?.replace('{review}', reviewText);
    }
  } else {
    // センチメント分析等はデフォルトプロンプト
    prompt = PROMPTS[type]?.replace('{review}', reviewText);
  }
  
  if (!prompt) {
    throw new Error(`Unknown request type: ${type}`);
  }
  
  console.log(`🤖 Generated prompt preview: ${prompt.substring(0, 200)}...`);

  console.log(`🤖 OpenAI API request - Type: ${type}, Length: ${reviewText.length}`);

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: MODEL_CONFIG[type] || 'gpt-4o-mini', // タイプ別モデル選択
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1, // 一貫性重視
      max_tokens: type === 'sentiment' ? 200 : 50
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    console.error('OpenAI API error response:', errorData);
    throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const result = data.choices[0]?.message?.content;

  if (!result) {
    throw new Error('Empty response from OpenAI API');
  }

  // レスポンス処理
  if (type === 'classification') {
    const category = result.trim();
    
    // カスタムカテゴリがある場合はそれで検証、なければデフォルトカテゴリで検証
    const validCategories = customCategories && customCategories.length > 0 ? customCategories : CATEGORIES;
    
    if (!validCategories.includes(category)) {
      console.warn(`Unknown category returned: ${category}, defaulting to ${validCategories[validCategories.length - 1]}`);
      return { 
        category: validCategories[validCategories.length - 1], // 最後のカテゴリ（通常は「その他」）
        confidence: 0.5,
        original_response: category
      };
    }
    
    return { 
      category: category,
      confidence: 0.9, // OpenAIは高信頼度
      model: MODEL_CONFIG[type] || 'gpt-4o-mini'
    };
    
  } else if (type === 'sentiment') {
    try {
      // JSON形式で返される想定
      const sentimentData = JSON.parse(result);
      
      // 英語→カタカナ変換
      const labelMapping = {
        'Positive': 'ポジティブ',
        'Negative': 'ネガティブ', 
        'Neutral': 'ニュートラル'
      };
      
      const originalLabel = sentimentData.label || 'Neutral';
      const japaneseLabel = labelMapping[originalLabel] || originalLabel;
      
      return {
        sentiment: japaneseLabel,
        score: sentimentData.score || 0,
        reason: sentimentData.reason || '分析結果なし',
        model: MODEL_CONFIG[type] || 'gpt-4o-mini'
      };
    } catch (parseError) {
      // JSON解析に失敗した場合の堅牢なフォールバック処理
      console.warn('Failed to parse sentiment JSON, trying enhanced text parsing');
      console.warn('Raw result:', result);
      
      // 🔧 改善されたテキスト解析処理
      try {
        // JSONの不完全な部分をクリーニング
        let cleanedResult = result.replace(/^[^{]*/, '').replace(/[^}]*$/, '');
        if (cleanedResult.startsWith('{') && cleanedResult.endsWith('}')) {
          const retryParsed = JSON.parse(cleanedResult);
          const originalLabel = retryParsed.label || 'Neutral';
          const labelMapping = {
            'Positive': 'ポジティブ',
            'Negative': 'ネガティブ', 
            'Neutral': 'ニュートラル'
          };
          
          return {
            sentiment: labelMapping[originalLabel] || originalLabel,
            score: retryParsed.score || 0,
            reason: retryParsed.reason || '分析結果なし',
            model: MODEL_CONFIG[type] || 'gpt-4o-mini',
            note: 'Cleaned JSON parsing used'
          };
        }
      } catch (retryError) {
        console.warn('JSON cleaning also failed, using regex fallback');
      }
      
      // 正規表現によるフォールバック解析
      const sentimentMatch = result.match(/"label":\s*"(Positive|Negative|Neutral)"/i);
      const scoreMatch = result.match(/"score":\s*(-?\d+)/);
      const reasonMatch = result.match(/"reason":\s*"([^"]+)"/);
      
      const sentiment = sentimentMatch ? sentimentMatch[1] : 'Neutral';
      const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;
      let reason = reasonMatch ? reasonMatch[1] : '解析エラー';
      
      // 不正な理由テキストをフィルタ
      if (reason.includes('"score"') || reason.includes('"Positive"') || reason.includes('"Negative"') || reason.includes('"Neutral"')) {
        reason = '分析完了';
      }
      
      // 英語→カタカナ変換
      const labelMapping = {
        'Positive': 'ポジティブ',
        'Negative': 'ネガティブ', 
        'Neutral': 'ニュートラル'
      };
      const japaneseLabel = labelMapping[sentiment] || sentiment;
      
      return {
        sentiment: japaneseLabel,
        score: score,
        reason: reason,
        model: MODEL_CONFIG[type] || 'gpt-4o-mini',
        note: 'Enhanced fallback parsing used'
      };
    }
  }
  
  throw new Error(`Unsupported processing type: ${type}`);
}