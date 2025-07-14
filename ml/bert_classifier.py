"""
BERT分類器モジュール
日本語BERTを使用したペットフードレビューの分類
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os
import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

from config import MODEL_CONFIG, DEVICE_CONFIG, EVAL_CONFIG, PATHS
from data_preprocessing import DataPreprocessor

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataset(Dataset):
    """レビューデータセットクラス"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier:
    """BERT分類器クラス"""
    
    def __init__(self, model_name: str = None, num_labels: int = None):
        self.model_name = model_name or MODEL_CONFIG['bert_model_name']
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and DEVICE_CONFIG['use_gpu'] else 'cpu')
        
        # ディレクトリ作成
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)
        
        logger.info(f"使用デバイス: {self.device}")
    
    def load_tokenizer(self):
        """トークナイザーの読み込み"""
        logger.info(f"トークナイザーを読み込み中: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # トークナイザーにパディングトークンが設定されていない場合は設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_model(self, num_labels: int):
        """モデルの読み込み"""
        logger.info(f"モデルを読み込み中: {self.model_name}")
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
    
    def prepare_data(self, processed_data: Dict) -> Tuple:
        """
        前処理済みデータから学習用データを準備
        
        Args:
            processed_data: 前処理済みデータ
            
        Returns:
            Tuple: (train_dataset, val_dataset, test_dataset)
        """
        logger.info("学習用データを準備中...")
        
        # データ分割の取得
        split_data = processed_data['split_data']
        self.label_encoder = processed_data['label_encoder']
        
        if len(split_data) == 6:  # train, val, test
            X_train, X_val, X_test, y_train, y_val, y_test = split_data
        else:  # train, test only
            X_train, X_test, y_train, y_test = split_data
            X_val, y_val = None, None
        
        # トークナイザーの準備
        if self.tokenizer is None:
            self.load_tokenizer()
        
        # データセットの作成
        train_dataset = ReviewDataset(
            X_train, y_train, self.tokenizer, MODEL_CONFIG['max_length']
        )
        
        test_dataset = ReviewDataset(
            X_test, y_test, self.tokenizer, MODEL_CONFIG['max_length']
        )
        
        val_dataset = None
        if X_val is not None:
            val_dataset = ReviewDataset(
                X_val, y_val, self.tokenizer, MODEL_CONFIG['max_length']
            )
        
        logger.info(f"学習データ: {len(train_dataset)}件")
        if val_dataset:
            logger.info(f"検証データ: {len(val_dataset)}件")
        logger.info(f"テストデータ: {len(test_dataset)}件")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """
        クラス重みの計算（不均衡データ対応）
        
        Args:
            labels: ラベル配列
            
        Returns:
            torch.Tensor: クラス重み
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        logger.info(f"クラス重み: {class_weights}")
        
        return class_weights_tensor
    
    def compute_metrics(self, eval_pred):
        """評価メトリクスの計算"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, processed_data: Dict, epochs: int = None, 
              use_class_weights: bool = True) -> Dict:
        """
        モデルの学習
        
        Args:
            processed_data: 前処理済みデータ
            epochs: エポック数
            use_class_weights: クラス重みを使用するか
            
        Returns:
            Dict: 学習結果
        """
        logger.info("=== モデル学習開始 ===")
        
        epochs = epochs or MODEL_CONFIG['num_epochs']
        
        # データ準備
        train_dataset, val_dataset, test_dataset = self.prepare_data(processed_data)
        
        # モデル読み込み
        num_labels = len(self.label_encoder.classes_)
        self.load_model(num_labels)
        
        # クラス重みの設定
        if use_class_weights:
            all_labels = processed_data['labels']
            class_weights = self.compute_class_weights(all_labels)
            
            # 損失関数にクラス重みを設定
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            self.model.config.loss_fn = loss_fn
        
        # 学習設定
        training_args = TrainingArguments(
            output_dir=PATHS['results_dir'],
            num_train_epochs=epochs,
            per_device_train_batch_size=MODEL_CONFIG['batch_size'],
            per_device_eval_batch_size=MODEL_CONFIG['batch_size'],
            learning_rate=MODEL_CONFIG['learning_rate'],
            weight_decay=MODEL_CONFIG['weight_decay'],
            warmup_steps=MODEL_CONFIG['warmup_steps'],
            logging_dir=PATHS['logs_dir'],
            logging_steps=50,
            evaluation_strategy="epoch" if val_dataset else "no",
            eval_steps=MODEL_CONFIG['eval_steps'] if val_dataset else None,
            save_strategy="epoch",
            save_steps=MODEL_CONFIG['save_steps'],
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_f1" if val_dataset else None,
            greater_is_better=True,
            save_total_limit=3,
            dataloader_num_workers=0,  # WindowsのマルチプロセシングIssue回避
            report_to=None,  # wandb等の外部ツールを無効化
        )
        
        # コールバック設定
        callbacks = []
        if val_dataset:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
            callbacks.append(early_stopping)
        
        # Trainer初期化
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # 学習実行
        logger.info("学習を開始...")
        train_result = self.trainer.train()
        
        # モデル保存
        self.save_model()
        
        # テストデータで評価
        logger.info("テストデータで評価中...")
        test_results = self.trainer.evaluate(test_dataset)
        
        # 詳細な評価結果を生成
        test_predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(test_predictions.predictions, axis=1)
        y_true = test_predictions.label_ids
        
        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        
        # クラス別の詳細な評価
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        class_report = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_report[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        # 結果をまとめる
        results = {
            'train_result': train_result,
            'test_results': test_results,
            'confusion_matrix': cm.tolist(),
            'class_report': class_report,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'probabilities': test_predictions.predictions.tolist()
            },
            'model_config': {
                'model_name': self.model_name,
                'num_labels': num_labels,
                'num_epochs': epochs,
                'batch_size': MODEL_CONFIG['batch_size'],
                'learning_rate': MODEL_CONFIG['learning_rate']
            }
        }
        
        # 結果を保存
        self.save_results(results)
        
        logger.info("=== モデル学習完了 ===")
        logger.info(f"テスト精度: {test_results['eval_accuracy']:.4f}")
        logger.info(f"テストF1: {test_results['eval_f1']:.4f}")
        
        return results
    
    def predict(self, text: str, return_probabilities: bool = False) -> Dict:
        """
        単一テキストの予測
        
        Args:
            text: 予測するテキスト
            return_probabilities: 確率を返すか
            
        Returns:
            Dict: 予測結果
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("モデルまたはトークナイザーが読み込まれていません")
        
        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MODEL_CONFIG['max_length'],
            return_tensors='pt'
        )
        
        # デバイスに移動
        encoding = {key: val.to(self.device) for key, val in encoding.items()}
        
        # 予測実行
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 結果の処理
        predicted_class_id = predictions.argmax().item()
        confidence = predictions.max().item()
        predicted_label = self.label_encoder.inverse_transform([predicted_class_id])[0]
        
        result = {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'predicted_class_id': predicted_class_id
        }
        
        if return_probabilities:
            all_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                all_probabilities[class_name] = predictions[0][i].item()
            result['all_probabilities'] = all_probabilities
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        バッチ予測
        
        Args:
            texts: 予測するテキストリスト
            
        Returns:
            List[Dict]: 予測結果リスト
        """
        # データセット作成（ダミーラベル）
        dummy_labels = [0] * len(texts)
        dataset = ReviewDataset(texts, dummy_labels, self.tokenizer, MODEL_CONFIG['max_length'])
        
        # DataLoader作成
        dataloader = DataLoader(dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=False)
        
        # 予測実行
        all_predictions = []
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="予測中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
        
        # 結果の整形
        results = []
        for i, text in enumerate(texts):
            probs = all_predictions[i]
            predicted_class_id = np.argmax(probs)
            confidence = probs[predicted_class_id]
            predicted_label = self.label_encoder.inverse_transform([predicted_class_id])[0]
            
            result = {
                'text': text,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'predicted_class_id': predicted_class_id
            }
            results.append(result)
        
        return results
    
    def save_model(self, save_path: str = None):
        """モデルとトークナイザーの保存"""
        if save_path is None:
            save_path = PATHS['models_dir']
        
        os.makedirs(save_path, exist_ok=True)
        
        # モデル保存
        model_path = os.path.join(save_path, 'model')
        self.model.save_pretrained(model_path)
        logger.info(f"モデルを保存: {model_path}")
        
        # トークナイザー保存
        tokenizer_path = os.path.join(save_path, 'tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"トークナイザーを保存: {tokenizer_path}")
        
        # ラベルエンコーダー保存
        le_path = os.path.join(save_path, 'label_encoder.pkl')
        with open(le_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"ラベルエンコーダーを保存: {le_path}")
    
    def load_saved_model(self, model_path: str = None):
        """保存されたモデルの読み込み"""
        if model_path is None:
            model_path = PATHS['models_dir']
        
        # トークナイザー読み込み
        tokenizer_path = os.path.join(model_path, 'tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # ラベルエンコーダー読み込み
        le_path = os.path.join(model_path, 'label_encoder.pkl')
        with open(le_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # モデル読み込み
        model_dir = os.path.join(model_path, 'model')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        
        logger.info(f"保存されたモデルを読み込み: {model_path}")
    
    def save_results(self, results: Dict, filename: str = 'training_results.json'):
        """学習結果の保存"""
        results_path = os.path.join(PATHS['results_dir'], filename)
        
        # numpy配列をリストに変換してJSON保存可能にする
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # 再帰的にnumpy配列を変換
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {key: recursive_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"学習結果を保存: {results_path}")

if __name__ == "__main__":
    # 使用例
    from data_preprocessing import DataPreprocessor
    
    # データ前処理
    preprocessor = DataPreprocessor()
    excel_file = "../ラベル付け参考用.xlsx"
    
    if os.path.exists(excel_file):
        processed_data = preprocessor.process_data(excel_file)
        
        # BERT分類器で学習
        classifier = BERTClassifier()
        results = classifier.train(processed_data, epochs=3)
        
        # テスト予測
        test_text = """
        今までドライフードを色々試してきましたが、どれもちゃんと食べずに残してました。
        こちらは初めて与えた瞬間から飛び付いて喜んで完食してくれました。
        それからご飯の時間になると、催促するようになるまでになりました！
        うんちの匂いもかなり減った気がします^_^
        """
        
        prediction = classifier.predict(test_text, return_probabilities=True)
        print(f"\\n予測結果: {prediction['predicted_label']}")
        print(f"確信度: {prediction['confidence']:.3f}")
        
    else:
        print(f"ファイルが見つかりません: {excel_file}")
