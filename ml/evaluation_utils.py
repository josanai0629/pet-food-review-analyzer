"""
評価・可視化ユーティリティモジュール
学習結果の詳細分析と可視化を行う
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

from config import PATHS, EDA_CONFIG

# ログ設定
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or PATHS['figures_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 日本語フォント設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo']
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], normalize: bool = True,
                            save_path: str = None) -> None:
        """
        混同行列の可視化
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            class_names: クラス名のリスト
            normalize: 正規化するかどうか
            save_path: 保存パス
        """
        # 混同行列の計算
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = '正規化混同行列'
            fmt = '.2%'
            cm_display = cm_norm
        else:
            title = '混同行列'
            fmt = 'd'
            cm_display = cm
        
        # プロット作成
        plt.figure(figsize=(12, 10))
        
        # ヒートマップ
        sns.heatmap(cm_display, 
                   annot=True, 
                   fmt=fmt,
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': '割合' if normalize else '件数'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('予測ラベル', fontsize=12)
        plt.ylabel('実際のラベル', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 各セルに件数も表示（正規化の場合）
        if normalize:
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j+0.5, i+0.7, f'({cm[i,j]})', 
                           ha='center', va='center', fontsize=8, color='gray')
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"混同行列を保存: {save_path}")
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: List[str], save_path: str = None) -> Dict:
        """
        分類レポートの可視化
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            class_names: クラス名のリスト
            save_path: 保存パス
            
        Returns:
            Dict: 分類レポート
        """
        # 分類レポートの計算
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # DataFrameに変換
        df_report = pd.DataFrame(report).transpose()
        
        # 可視化用のデータ準備
        metrics_df = df_report.iloc[:-3, :-1]  # macro avg, weighted avg, accuracyを除く
        
        # ヒートマップ作成
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(metrics_df.values, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   xticklabels=['Precision', 'Recall', 'F1-Score'],
                   yticklabels=metrics_df.index,
                   cbar_kws={'label': 'スコア'},
                   vmin=0, vmax=1)
        
        plt.title('クラス別評価メトリクス', fontsize=16, fontweight='bold')
        plt.xlabel('メトリクス', fontsize=12)
        plt.ylabel('クラス', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'classification_report.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 全体的な統計の表示
        print("\\n=== 分類レポート ===")
        print(f"全体精度: {report['accuracy']:.3f}")
        print(f"マクロ平均 F1: {report['macro avg']['f1-score']:.3f}")
        print(f"重み付き平均 F1: {report['weighted avg']['f1-score']:.3f}")
        
        logger.info(f"分類レポートを保存: {save_path}")
        
        return report
    
    def plot_learning_curves(self, training_history: Dict, save_path: str = None) -> None:
        """
        学習曲線の可視化
        
        Args:
            training_history: 学習履歴
            save_path: 保存パス
        """
        if 'log_history' not in training_history:
            logger.warning("学習履歴が見つかりません")
            return
        
        log_history = training_history['log_history']
        
        # データの抽出
        epochs = []
        train_loss = []
        eval_loss = []
        eval_f1 = []
        
        for entry in log_history:
            if 'epoch' in entry:
                epochs.append(entry['epoch'])
                if 'train_loss' in entry:
                    train_loss.append(entry['train_loss'])
                if 'eval_loss' in entry:
                    eval_loss.append(entry['eval_loss'])
                if 'eval_f1' in entry:
                    eval_f1.append(entry['eval_f1'])
        
        # プロット作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 損失の推移
        if train_loss and eval_loss:
            ax1.plot(epochs[:len(train_loss)], train_loss, 'b-', label='訓練損失', linewidth=2)
            ax1.plot(epochs[:len(eval_loss)], eval_loss, 'r-', label='検証損失', linewidth=2)
            ax1.set_xlabel('エポック')
            ax1.set_ylabel('損失')
            ax1.set_title('学習曲線（損失）')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # F1スコアの推移
        if eval_f1:
            ax2.plot(epochs[:len(eval_f1)], eval_f1, 'g-', label='検証F1', linewidth=2)
            ax2.set_xlabel('エポック')
            ax2.set_ylabel('F1スコア')
            ax2.set_title('学習曲線（F1スコア）')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'learning_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"学習曲線を保存: {save_path}")
    
    def plot_prediction_confidence_distribution(self, predictions: List[Dict],
                                              save_path: str = None) -> None:
        """
        予測確信度の分布を可視化
        
        Args:
            predictions: 予測結果のリスト
            save_path: 保存パス
        """
        # 確信度とラベルを抽出
        confidences = [pred['confidence'] for pred in predictions]
        labels = [pred['predicted_label'] for pred in predictions]
        is_correct = [pred.get('correct', True) for pred in predictions]
        
        # プロット作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 確信度のヒストグラム
        ax1.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均: {np.mean(confidences):.3f}')
        ax1.set_xlabel('確信度')
        ax1.set_ylabel('頻度')
        ax1.set_title('予測確信度の分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 正解/不正解別の確信度分布
        if any(isinstance(c, bool) for c in is_correct):
            correct_conf = [conf for conf, correct in zip(confidences, is_correct) if correct]
            incorrect_conf = [conf for conf, correct in zip(confidences, is_correct) if not correct]
            
            ax2.hist(correct_conf, bins=20, alpha=0.7, label='正解', color='green')
            ax2.hist(incorrect_conf, bins=20, alpha=0.7, label='不正解', color='red')
            ax2.set_xlabel('確信度')
            ax2.set_ylabel('頻度')
            ax2.set_title('正解/不正解別確信度分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confidence_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"確信度分布を保存: {save_path}")
    
    def create_interactive_results_dashboard(self, results: Dict, 
                                           save_path: str = None) -> str:
        """
        インタラクティブな結果ダッシュボードを作成
        
        Args:
            results: 学習結果
            save_path: 保存パス
            
        Returns:
            str: 保存されたHTMLファイルのパス
        """
        # データの準備
        test_results = results.get('test_results', {})
        class_report = results.get('class_report', {})
        
        # サブプロットの作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('全体メトリクス', 'クラス別F1スコア', 
                          'クラス別精度', 'クラス別再現率'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 全体メトリクス（ゲージチャート）
        overall_f1 = test_results.get('eval_f1', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_f1,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "全体F1スコア"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=1, col=1
        )
        
        # クラス別メトリクス
        if class_report:
            class_names = []
            f1_scores = []
            precisions = []
            recalls = []
            
            for class_name, metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    class_names.append(class_name)
                    f1_scores.append(metrics.get('f1', 0))
                    precisions.append(metrics.get('precision', 0))
                    recalls.append(metrics.get('recall', 0))
            
            # F1スコア
            fig.add_trace(
                go.Bar(x=class_names, y=f1_scores, name='F1スコア',
                      marker_color='blue'),
                row=1, col=2
            )
            
            # 精度
            fig.add_trace(
                go.Bar(x=class_names, y=precisions, name='精度',
                      marker_color='green'),
                row=2, col=1
            )
            
            # 再現率
            fig.add_trace(
                go.Bar(x=class_names, y=recalls, name='再現率',
                      marker_color='orange'),
                row=2, col=2
            )
        
        # レイアウト更新
        fig.update_layout(
            title="モデル評価ダッシュボード",
            height=800,
            showlegend=False
        )
        
        # 軸のラベル設定
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(range=[0, 1])
        
        # 保存
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'results_dashboard.html')
        
        fig.write_html(save_path)
        
        logger.info(f"インタラクティブダッシュボードを保存: {save_path}")
        
        return save_path
    
    def generate_evaluation_report(self, results: Dict, output_path: str = None) -> str:
        """
        総合評価レポートの生成
        
        Args:
            results: 学習結果
            output_path: 出力パス
            
        Returns:
            str: レポートファイルのパス
        """
        if output_path is None:
            output_path = os.path.join(self.save_dir, 'evaluation_report.md')
        
        # レポートの作成
        report_lines = []
        report_lines.append("# ペットフードレビュー分類器 - 評価レポート\\n")
        
        # 実験情報
        if 'model_config' in results:
            config = results['model_config']
            report_lines.append("## 実験設定\\n")
            report_lines.append(f"- **モデル**: {config.get('model_name', 'N/A')}")
            report_lines.append(f"- **エポック数**: {config.get('num_epochs', 'N/A')}")
            report_lines.append(f"- **バッチサイズ**: {config.get('batch_size', 'N/A')}")
            report_lines.append(f"- **学習率**: {config.get('learning_rate', 'N/A')}")
            report_lines.append("\\n")
        
        # 全体性能
        if 'test_results' in results:
            test_results = results['test_results']
            report_lines.append("## 全体性能\\n")
            report_lines.append(f"- **テスト精度**: {test_results.get('eval_accuracy', 0):.4f}")
            report_lines.append(f"- **テストF1**: {test_results.get('eval_f1', 0):.4f}")
            report_lines.append(f"- **テスト精度**: {test_results.get('eval_precision', 0):.4f}")
            report_lines.append(f"- **テスト再現率**: {test_results.get('eval_recall', 0):.4f}")
            report_lines.append("\\n")
        
        # クラス別性能
        if 'class_report' in results:
            class_report = results['class_report']
            report_lines.append("## クラス別性能\\n")
            report_lines.append("| クラス | 精度 | 再現率 | F1スコア | サポート |")
            report_lines.append("|--------|------|--------|----------|----------|")
            
            for class_name, metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    f1 = metrics.get('f1', 0)
                    support = metrics.get('support', 0)
                    report_lines.append(f"| {class_name} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support} |")
            
            report_lines.append("\\n")
        
        # 予測例
        if 'evaluation' in results and 'test_cases' in results['evaluation']:
            test_cases = results['evaluation']['test_cases']
            report_lines.append("## 予測例\\n")
            
            for i, case in enumerate(test_cases[:5], 1):  # 最初の5件
                report_lines.append(f"### 例 {i}")
                report_lines.append(f"**テキスト**: {case.get('text', '')[:100]}...")
                report_lines.append(f"**予測**: {case.get('predicted', 'N/A')}")
                report_lines.append(f"**確信度**: {case.get('confidence', 0):.3f}")
                if 'expected' in case:
                    result = "✅ 正解" if case.get('correct', False) else "❌ 不正解"
                    report_lines.append(f"**結果**: {result}")
                report_lines.append("\\n")
        
        # 改善提案
        report_lines.append("## 改善提案\\n")
        report_lines.append("### 短期的改善")
        report_lines.append("- ハイパーパラメータの調整")
        report_lines.append("- データ前処理の最適化")
        report_lines.append("- クラス重みの調整\\n")
        
        report_lines.append("### 長期的改善")
        report_lines.append("- より多くのデータ収集")
        report_lines.append("- アンサンブル学習の導入")
        report_lines.append("- ドメイン特化型事前訓練")
        
        # ファイルに書き込み
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(report_lines))
        
        logger.info(f"評価レポートを保存: {output_path}")
        
        return output_path

if __name__ == "__main__":
    # 使用例
    evaluator = ModelEvaluator()
    
    # サンプルデータでテスト
    import json
    
    # 結果ファイルがあれば読み込んで可視化
    results_file = "results/training_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 各種可視化の実行
        print("評価・可視化を実行中...")
        
        # ダッシュボード作成
        dashboard_path = evaluator.create_interactive_results_dashboard(results)
        print(f"ダッシュボード: {dashboard_path}")
        
        # レポート生成
        report_path = evaluator.generate_evaluation_report(results)
        print(f"レポート: {report_path}")
        
    else:
        print(f"結果ファイルが見つかりません: {results_file}")
