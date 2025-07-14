"""
学習実行スクリプト
データ前処理からBERT学習まで一括実行
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime
import torch

# 現在のディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_preprocessing import DataPreprocessor
from eda import EDAAnalyzer
from bert_classifier import BERTClassifier
from config import MODEL_CONFIG, DATA_CONFIG, PATHS

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PATHS['logs_dir'], f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """学習パイプラインクラス"""
    
    def __init__(self, config_overrides: dict = None):
        self.config_overrides = config_overrides or {}
        self.results = {}
        
        # ディレクトリ作成
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)
    
    def run_preprocessing(self, excel_file_path: str, run_eda: bool = True) -> dict:
        """
        データ前処理とEDAの実行
        
        Args:
            excel_file_path: Excelファイルのパス
            run_eda: EDAを実行するかどうか
            
        Returns:
            dict: 前処理済みデータ
        """
        logger.info("=== データ前処理開始 ===")
        
        # データ前処理
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_data(excel_file_path)
        
        # 統計情報の表示
        logger.info(f"総データ数: {len(processed_data['texts'])}")
        logger.info(f"ラベル数: {len(processed_data['label_encoder'].classes_)}")
        logger.info(f"ラベル一覧: {list(processed_data['label_encoder'].classes_)}")
        
        # EDAの実行
        if run_eda:
            logger.info("=== EDA分析開始 ===")
            eda_analyzer = EDAAnalyzer()
            eda_results = eda_analyzer.run_complete_eda(excel_file_path)
            processed_data['eda_results'] = eda_results
        
        # 前処理済みデータの保存
        preprocessor.save_processed_data(processed_data)
        
        self.results['preprocessing'] = {
            'data_stats': processed_data['data_stats'],
            'label_analysis': processed_data['label_analysis']
        }
        
        return processed_data
    
    def run_training(self, processed_data: dict, 
                    epochs: int = None, 
                    use_class_weights: bool = True,
                    model_name: str = None) -> dict:
        """
        BERT学習の実行
        
        Args:
            processed_data: 前処理済みデータ
            epochs: エポック数
            use_class_weights: クラス重みを使用するか
            model_name: 使用するモデル名
            
        Returns:
            dict: 学習結果
        """
        logger.info("=== BERT学習開始 ===")
        
        # パラメータの設定
        epochs = epochs or self.config_overrides.get('epochs', MODEL_CONFIG['num_epochs'])
        model_name = model_name or self.config_overrides.get('model_name', MODEL_CONFIG['bert_model_name'])
        
        # 分類器の初期化
        classifier = BERTClassifier(model_name=model_name)
        
        # 学習実行
        training_results = classifier.train(
            processed_data=processed_data,
            epochs=epochs,
            use_class_weights=use_class_weights
        )
        
        # 結果の保存
        self.results['training'] = training_results
        
        # モデル情報を結果に追加
        training_results['model_info'] = {
            'model_name': model_name,
            'total_parameters': sum(p.numel() for p in classifier.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in classifier.model.parameters() if p.requires_grad),
            'device': str(classifier.device)
        }
        
        logger.info(f"学習完了 - テスト精度: {training_results['test_results']['eval_accuracy']:.4f}")
        
        return training_results, classifier
    
    def evaluate_model(self, classifier: BERTClassifier, test_cases: list = None) -> dict:
        """
        モデルの詳細評価
        
        Args:
            classifier: 学習済み分類器
            test_cases: テストケース
            
        Returns:
            dict: 評価結果
        """
        logger.info("=== モデル評価開始 ===")
        
        # デフォルトのテストケース
        if test_cases is None:
            test_cases = [
                {
                    'text': '今までドライフードを色々試してきましたが、どれもちゃんと食べずに残してました。こちらは初めて与えた瞬間から飛び付いて喜んで完食してくれました。',
                    'expected': '食べる'
                },
                {
                    'text': 'うちの猫はこれを食べると、下痢でした',
                    'expected': '吐く・便が悪くなる'
                },
                {
                    'text': '配送の箱が破れていて中身がこぼれていました',
                    'expected': '配送・梱包'
                },
                {
                    'text': '値段が高くなってきて困ります',
                    'expected': '値上がり/高い'
                },
                {
                    'text': '安くて助かります',
                    'expected': '安い'
                }
            ]
        
        evaluation_results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                prediction = classifier.predict(
                    test_case['text'], 
                    return_probabilities=True
                )
                
                result = {
                    'test_id': i + 1,
                    'text': test_case['text'],
                    'expected': test_case.get('expected', 'Unknown'),
                    'predicted': prediction['predicted_label'],
                    'confidence': prediction['confidence'],
                    'correct': prediction['predicted_label'] == test_case.get('expected'),
                    'all_probabilities': prediction.get('all_probabilities', {})
                }
                
                evaluation_results.append(result)
                
                logger.info(f"テスト{i+1}: {result['predicted']} (確信度: {result['confidence']:.3f}) "
                          f"{'✓' if result['correct'] else '✗'}")
                
            except Exception as e:
                logger.error(f"テスト{i+1}でエラー: {e}")
                evaluation_results.append({
                    'test_id': i + 1,
                    'text': test_case['text'],
                    'error': str(e)
                })
        
        # 正解率計算
        correct_predictions = sum(1 for r in evaluation_results if r.get('correct', False))
        accuracy = correct_predictions / len(evaluation_results) if evaluation_results else 0
        
        eval_summary = {
            'test_cases': evaluation_results,
            'accuracy': accuracy,
            'total_tests': len(evaluation_results),
            'correct_predictions': correct_predictions
        }
        
        logger.info(f"テストケース正解率: {accuracy:.3f} ({correct_predictions}/{len(evaluation_results)})")
        
        return eval_summary
    
    def save_experiment_results(self, experiment_name: str = None):
        """実験結果の保存"""
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 実験結果をまとめる
        experiment_results = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'config_overrides': self.config_overrides,
                'model_config': MODEL_CONFIG,
                'data_config': DATA_CONFIG
            },
            'results': self.results
        }
        
        # 保存
        results_path = os.path.join(PATHS['results_dir'], f'{experiment_name}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"実験結果を保存: {results_path}")
        
        return experiment_results
    
    def run_complete_pipeline(self, excel_file_path: str, 
                            epochs: int = None,
                            run_eda: bool = True,
                            test_cases: list = None,
                            experiment_name: str = None) -> dict:
        """
        完全なパイプライン実行
        
        Args:
            excel_file_path: Excelファイルのパス
            epochs: エポック数
            run_eda: EDAを実行するか
            test_cases: テストケース
            experiment_name: 実験名
            
        Returns:
            dict: 完全な実験結果
        """
        logger.info("=== 完全パイプライン開始 ===")
        
        start_time = datetime.now()
        
        try:
            # 1. データ前処理とEDA
            processed_data = self.run_preprocessing(excel_file_path, run_eda)
            
            # 2. BERT学習
            training_results, classifier = self.run_training(
                processed_data, epochs=epochs
            )
            
            # 3. モデル評価
            evaluation_results = self.evaluate_model(classifier, test_cases)
            self.results['evaluation'] = evaluation_results
            
            # 4. 実験結果の保存
            experiment_results = self.save_experiment_results(experiment_name)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"=== パイプライン完了 ===")
            logger.info(f"実行時間: {duration}")
            logger.info(f"最終テスト精度: {training_results['test_results']['eval_accuracy']:.4f}")
            logger.info(f"テストケース正解率: {evaluation_results['accuracy']:.3f}")
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"パイプライン実行エラー: {e}")
            raise

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='ペットフードレビュー分類器の学習')
    
    parser.add_argument('--excel_file', type=str, required=True,
                       help='学習用Excelファイルのパス')
    parser.add_argument('--epochs', type=int, default=5,
                       help='エポック数 (デフォルト: 5)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='バッチサイズ (デフォルト: 8)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学習率 (デフォルト: 2e-5)')
    parser.add_argument('--model_name', type=str, 
                       default='tohoku-nlp/bert-base-japanese-whole-word-masking',
                       help='使用するBERTモデル名')
    parser.add_argument('--experiment_name', type=str,
                       help='実験名')
    parser.add_argument('--skip_eda', action='store_true',
                       help='EDAをスキップ')
    parser.add_argument('--gpu', action='store_true',
                       help='GPUを使用')
    
    args = parser.parse_args()
    
    # 設定の上書き
    config_overrides = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'model_name': args.model_name,
        'use_gpu': args.gpu
    }
    
    # MODEL_CONFIGを動的に更新
    if args.batch_size != MODEL_CONFIG['batch_size']:
        MODEL_CONFIG['batch_size'] = args.batch_size
    if args.learning_rate != MODEL_CONFIG['learning_rate']:
        MODEL_CONFIG['learning_rate'] = args.learning_rate
    
    # ファイル存在確認
    if not os.path.exists(args.excel_file):
        logger.error(f"ファイルが見つかりません: {args.excel_file}")
        return 1
    
    # GPU設定
    if args.gpu and torch.cuda.is_available():
        logger.info(f"GPU使用: {torch.cuda.get_device_name()}")
    else:
        logger.info("CPU使用")
    
    # パイプライン実行
    pipeline = TrainingPipeline(config_overrides)
    
    try:
        results = pipeline.run_complete_pipeline(
            excel_file_path=args.excel_file,
            epochs=args.epochs,
            run_eda=not args.skip_eda,
            experiment_name=args.experiment_name
        )
        
        print("\\n" + "="*50)
        print("🎉 学習完了！")
        print("="*50)
        print(f"実験名: {results['experiment_info']['name']}")
        print(f"テスト精度: {results['results']['training']['test_results']['eval_accuracy']:.4f}")
        print(f"テストF1: {results['results']['training']['test_results']['eval_f1']:.4f}")
        print(f"テストケース正解率: {results['results']['evaluation']['accuracy']:.3f}")
        print(f"結果保存先: {PATHS['results_dir']}")
        print(f"モデル保存先: {PATHS['models_dir']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"実行失敗: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
