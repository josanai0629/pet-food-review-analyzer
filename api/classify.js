/**
 * OpenAI API統合 - ラベル付けとセンチメント分析
 * Vercel Functions用エンドポイント
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

  const { review, type = 'classification', batchReviews, categories } = req.body;
  
  // 入力検証
  if (!review && !batchReviews) {
    return res.status(400).json({ error: 'Review text or batch reviews required' });
  }

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
      console.log(`🔄 Batch processing: ${batchReviews.length} reviews`);
      const results = [];
      
      for (let i = 0; i < batchReviews.length; i++) {
        const batchReview = batchReviews[i];
        try {
          const result = await processOpenAIRequest(batchReview.text, type, categories);
          results.push({
            id: batchReview.id || i,
            ...result
          });
          
          // レート制限対策：リクエスト間に短い待機
          if (i < batchReviews.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        } catch (error) {
          console.error(`Batch processing error for review ${i}:`, error);
          results.push({
            id: batchReview.id || i,
            error: 'Processing failed'
          });
        }
      }
      
      return res.status(200).json({ results });
    }

    // 単一レビューの処理
    const result = await processOpenAIRequest(review, type, categories);
    return res.status(200).json(result);

  } catch (error) {
    console.error('OpenAI API error:', error);
    return res.status(500).json({ 
      error: 'Classification failed',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
}

/**
 * OpenAI APIリクエスト処理（カスタムカテゴリ対応・固定テンプレート）
 */
async function processOpenAIRequest(reviewText, type, customCategories = null) {
  let prompt;
  
  if (type === 'classification') {
    if (customCategories && customCategories.length > 0) {
      // 固定プロンプトテンプレートにカスタムカテゴリを埋め込み
      const categoryList = customCategories.map((cat, index) => `${index + 1}. ${cat}`).join('\n');
      
      // 🎯 固定プロンプトテンプレート
      prompt = `あなたはレビューの分析者です。記載されているレビューについて以下のラベルに基づいてラベリングしてください。

カテゴリ：
${categoryList}

レビュー: "${reviewText}"

回答は必ずカテゴリ名のみを返してください（例：${customCategories[0]}）。`;
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
      model: 'gpt-4o-mini', // コスト最適化
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
      model: 'gpt-4o-mini'
    };
    
  } else if (type === 'sentiment') {
    try {
      // JSON形式で返される想定
      const sentimentData = JSON.parse(result);
      
      return {
        sentiment: sentimentData.label || 'Neutral',
        score: sentimentData.score || 0,
        reason: sentimentData.reason || '分析結果なし',
        model: 'gpt-4o-mini'
      };
    } catch (parseError) {
      // JSON解析に失敗した場合の処理
      console.warn('Failed to parse sentiment JSON, trying text parsing');
      
      // テキスト形式での解析を試行
      const lines = result.split('\n').filter(line => line.trim());
      const sentiment = lines.find(line => line.includes('sentiment'))?.split(':')[1]?.trim() || 'Neutral';
      const scoreMatch = result.match(/score[\":\s]*(-?\d+)/i);
      const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;
      const reason = lines.find(line => line.includes('reason'))?.split(':')[1]?.trim() || '解析エラー';
      
      return {
        sentiment: sentiment,
        score: score,
        reason: reason,
        model: 'gpt-4o-mini',
        note: 'Fallback parsing used'
      };
    }
  }
  
  throw new Error(`Unsupported processing type: ${type}`);
}