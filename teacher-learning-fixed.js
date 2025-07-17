/**
 * 🏆 修正版：高精度ペットフードレビュー教師学習システム
 * 教師データから学習して10カテゴリ分類精度を大幅向上
 */
class PetFoodTeacherLearning {
    constructor() {
        // 10固定カテゴリ
        this.categories = [
            '吐く・便が悪くなる',
            '食べない',
            '吐き戻し・便の改善',
            '食べる',
            '値上がり/高い',
            '安い',
            '配送・梱包',
            '賞味期限',
            'ジッパー',
            'その他'
        ];

        // 学習データストレージ
        this.categoryPatterns = {};
        this.wordScores = {};
        this.isLearned = false;
        this.accuracyHistory = [];
        
        // 各カテゴリ初期化
        this.categories.forEach(category => {
            this.categoryPatterns[category] = new Map();
            this.wordScores[category] = new Map();
        });
        
        // モデル読み込み
        this.loadModel();
    }
    
    /**
     * 教師データからパターンを学習
     */
    processTeacherData(teacherData) {
        console.log('🎓 教師データ学習開始:', teacherData.length, '件');
        
        // カテゴリ分布確認
        const categoryStats = {};
        this.categories.forEach(cat => categoryStats[cat] = 0);
        teacherData.forEach(ex => {
            if (categoryStats[ex.category] !== undefined) {
                categoryStats[ex.category]++;
            }
        });
        console.log('📊 カテゴリ分布:', categoryStats);
        
        // 各カテゴリの特徴語を学習
        this.categories.forEach(category => {
            const examples = teacherData.filter(ex => ex.category === category);
            this.learnCategoryFeatures(category, examples, teacherData.length);
        });
        
        this.isLearned = true;
        this.saveModel();
        console.log('✅ 教師データ学習完了');
    }
    
    /**
     * カテゴリ特徴語学習
     */
    learnCategoryFeatures(category, examples, totalExamples) {
        if (examples.length === 0) return;
        
        const wordFreq = new Map();
        const totalWords = new Map();
        
        // 単語頻度計算
        examples.forEach(ex => {
            const words = this.extractWords(ex.text);
            words.forEach(word => {
                wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
                totalWords.set(word, (totalWords.get(word) || 0) + 1);
            });
        });
        
        // TF-IDF風スコア計算
        wordFreq.forEach((freq, word) => {
            const tf = freq / examples.length;
            const categoryOccurrence = this.categories.reduce((count, cat) => {
                return count + (this.wordScores[cat].has(word) ? 1 : 0);
            }, 0);
            const idf = Math.log(this.categories.length / (1 + categoryOccurrence));
            const score = tf * idf;
            
            if (score > 0.1) { // 閾値以上のみ保存
                this.wordScores[category].set(word, score);
            }
        });
        
        // 特徴的なフレーズパターンを学習
        this.learnPhrasePatterns(category, examples);
        
        console.log(`📝 ${category}: ${this.wordScores[category].size}語を学習`);
    }
    
    /**
     * フレーズパターン学習
     */
    learnPhrasePatterns(category, examples) {
        const patterns = new Map();
        
        // カテゴリ特有のフレーズを学習
        examples.forEach(ex => {
            const phrases = this.extractPhrases(ex.text, category);
            phrases.forEach(phrase => {
                patterns.set(phrase, (patterns.get(phrase) || 0) + 1);
            });
        });
        
        // 高頻度フレーズのみ保存
        patterns.forEach((freq, phrase) => {
            if (freq >= 2) {
                this.categoryPatterns[category].set(phrase, freq / examples.length);
            }
        });
    }
    
    /**
     * 単語抽出（日本語特化）
     */
    extractWords(text) {
        const words = [];
        const cleanText = text.toLowerCase().replace(/[！？。、]/g, '');
        
        // 日本語の単語パターンを抽出
        const patterns = [
            /[あ-んア-ンー・]+/g,  // ひらがな・カタカナ
            /[一-龯]+/g,           // 漢字
            /[a-zA-Z]{2,}/g        // 英語
        ];
        
        patterns.forEach(pattern => {
            const matches = cleanText.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    if (match.length >= 2 && !this.isStopWord(match)) {
                        words.push(match);
                    }
                });
            }
        });
        
        return words;
    }
    
    /**
     * フレーズ抽出（カテゴリ特化）
     */
    extractPhrases(text, category) {
        const phrases = [];
        const lowerText = text.toLowerCase();
        
        // カテゴリ特有のフレーズパターン
        const categoryPhrases = {
            '食べる': [
                /食いつき.{0,3}良/g, /よく食べ/g, /美味し/g, /完食/g, /喜んで食べ/g
            ],
            '食べない': [
                /食べ.{0,3}ない/g, /食いつき.{0,3}悪/g, /残し/g, /拒否/g, /見向きもしない/g
            ],
            '吐く・便が悪くなる': [
                /吐く/g, /下痢/g, /軟便/g, /体調不良/g, /お腹を壊/g
            ],
            '吐き戻し・便の改善': [
                /吐.{0,5}なくなった/g, /便.{0,5}良くなった/g, /調子が良/g, /元気になった/g
            ],
            '配送・梱包': [
                /配送/g, /梱包/g, /届いた/g, /箱/g, /破損/g
            ],
            '賞味期限': [
                /賞味期限/g, /期限/g, /日付/g
            ],
            'ジッパー': [
                /ジッパー/g, /チャック/g, /密封/g
            ],
            '値上がり/高い': [
                /値上がり/g, /高い/g, /値段が上が/g
            ],
            '安い': [
                /安い/g, /お得/g, /コスパ/g
            ]
        };
        
        if (categoryPhrases[category]) {
            categoryPhrases[category].forEach(pattern => {
                const matches = lowerText.match(pattern);
                if (matches) {
                    matches.forEach(match => phrases.push(match));
                }
            });
        }
        
        return phrases;
    }
    
    /**
     * ストップワード判定
     */
    isStopWord(word) {
        const stopWords = ['です', 'した', 'して', 'ある', 'いる', 'この', 'その', 'それ', 'から', 'まで', 'など'];
        return stopWords.includes(word) || word.length < 2;
    }
    
    /**
     * テキスト分類（学習後）
     */
    classifyText(text) {
        if (!this.isLearned) {
            return { category: 'その他', score: 0, confidence: 0 };
        }
        
        const scores = {};
        this.categories.forEach(cat => scores[cat] = 0);
        
        const words = this.extractWords(text);
        
        // 学習した単語スコアを適用
        this.categories.forEach(category => {
            let categoryScore = 0;
            
            // 単語スコア
            words.forEach(word => {
                const wordScore = this.wordScores[category].get(word) || 0;
                categoryScore += wordScore * 10;
            });
            
            // フレーズパターンスコア
            const phrases = this.extractPhrases(text, category);
            phrases.forEach(phrase => {
                const phraseScore = this.categoryPatterns[category].get(phrase) || 0;
                categoryScore += phraseScore * 15;
            });
            
            scores[category] = categoryScore;
        });
        
        // 最高スコアのカテゴリを選択
        let bestCategory = 'その他';
        let bestScore = 0;
        
        Object.entries(scores).forEach(([category, score]) => {
            if (score > bestScore) {
                bestScore = score;
                bestCategory = category;
            }
        });
        
        // 信頼度計算
        const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
        const confidence = totalScore > 0 ? bestScore / totalScore : 0;
        
        return {
            category: bestCategory,
            score: bestScore,
            confidence: confidence,
            allScores: scores
        };
    }
    
    /**
     * ユーザーフィードバックから学習
     */
    learnFromFeedback(text, correctCategory, predictedCategory) {
        if (!this.categories.includes(correctCategory)) return;
        
        // 正解カテゴリの特徴語を強化
        const words = this.extractWords(text);
        words.forEach(word => {
            const currentScore = this.wordScores[correctCategory].get(word) || 0;
            this.wordScores[correctCategory].set(word, currentScore + 0.1);
        });
        
        // 誤答カテゴリの特徴語を減点
        if (predictedCategory !== correctCategory && predictedCategory !== 'その他') {
            words.forEach(word => {
                const currentScore = this.wordScores[predictedCategory].get(word) || 0;
                this.wordScores[predictedCategory].set(word, Math.max(0, currentScore - 0.05));
            });
        }
        
        this.saveModel();
    }
    
    /**
     * モデル保存
     */
    saveModel() {
        const model = {
            isLearned: this.isLearned,
            wordScores: this.serializeMapStructure(this.wordScores),
            categoryPatterns: this.serializeMapStructure(this.categoryPatterns),
            accuracyHistory: this.accuracyHistory,
            timestamp: new Date().toISOString()
        };
        
        localStorage.setItem('petFoodTeacherModel', JSON.stringify(model));
    }
    
    /**
     * モデル読み込み
     */
    loadModel() {
        const savedModel = localStorage.getItem('petFoodTeacherModel');
        if (!savedModel) return;
        
        try {
            const model = JSON.parse(savedModel);
            
            this.isLearned = model.isLearned || false;
            this.accuracyHistory = model.accuracyHistory || [];
            
            if (model.wordScores) {
                this.wordScores = this.deserializeMapStructure(model.wordScores);
            }
            
            if (model.categoryPatterns) {
                this.categoryPatterns = this.deserializeMapStructure(model.categoryPatterns);
            }
            
            if (this.isLearned) {
                console.log('✅ 学習済みモデルを読み込みました');
            }
        } catch (e) {
            console.error('❌ モデル読み込みエラー:', e);
        }
    }
    
    /**
     * Map構造のシリアライズ
     */
    serializeMapStructure(mapStructure) {
        const serialized = {};
        Object.keys(mapStructure).forEach(key => {
            serialized[key] = Object.fromEntries(mapStructure[key]);
        });
        return serialized;
    }
    
    /**
     * Map構造のデシリアライズ
     */
    deserializeMapStructure(serialized) {
        const mapStructure = {};
        Object.keys(serialized).forEach(key => {
            mapStructure[key] = new Map(Object.entries(serialized[key]));
        });
        return mapStructure;
    }
    
    /**
     * モデルリセット
     */
    resetModel() {
        this.isLearned = false;
        this.accuracyHistory = [];
        this.categories.forEach(category => {
            this.wordScores[category].clear();
            this.categoryPatterns[category].clear();
        });
        localStorage.removeItem('petFoodTeacherModel');
    }
}