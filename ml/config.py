"""
機械学習モジュールの設定ファイル
"""
import os

# モデル設定
MODEL_CONFIG = {
    'bert_model_name': 'tohoku-nlp/bert-base-japanese-whole-word-masking',
    'max_length': 512,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'eval_steps': 100,
    'save_steps': 500,
}

# データ設定
DATA_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_seed': 42,
    'min_text_length': 5,  # 最小文字数
    'max_text_length': 1000,  # 最大文字数
}

# パス設定
PATHS = {
    'data_dir': '../data',
    'models_dir': './models',
    'results_dir': './results',
    'logs_dir': './logs',
    'figures_dir': './figures',
}

# ログ設定
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(PATHS['logs_dir'], 'training.log'),
}

# GPU設定
DEVICE_CONFIG = {
    'use_gpu': True,
    'gpu_id': 0,
}

# 評価設定
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'average': 'weighted',  # macro, micro, weighted
}

# ラベル設定（データから動的に取得するため、初期値として設定）
LABEL_CONFIG = {
    'target_labels': [
        '食べる', '食べない', '吐き戻し・便の改善', 'その他',
        '安い', '配送・梱包', '値上がり/高い', '吐く・便が悪くなる',
        '賞味期限', 'ジッパー'
    ],
    'label_weights': None,  # クラス不均衡対応用（動的に計算）
}

# 前処理設定
PREPROCESSING_CONFIG = {
    'normalize_text': True,
    'remove_urls': True,
    'remove_emails': True,
    'remove_extra_whitespace': True,
    'lowercase': False,  # 日本語なのでFalse
}

# EDA設定
EDA_CONFIG = {
    'max_wordcloud_words': 100,
    'wordcloud_width': 800,
    'wordcloud_height': 400,
    'figure_size': (12, 8),
    'sample_size_for_display': 5,  # 各ラベルのサンプル表示数
}

# 実験設定
EXPERIMENT_CONFIG = {
    'experiment_name': 'pet_food_review_classification',
    'save_model': True,
    'save_tokenizer': True,
    'save_predictions': True,
    'cross_validation': False,  # k-fold cross validation
    'cv_folds': 5,
}
