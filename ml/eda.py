"""
探索的データ分析（EDA）モジュール
ペットフードレビューデータの可視化と分析を行う
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re

from config import EDA_CONFIG, PATHS
from data_preprocessing import DataPreprocessor

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDAAnalyzer:
    """探索的データ分析クラス"""
    
    def __init__(self):
        self.figures_dir = PATHS['figures_dir']
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # カラーパレット設定
        self.colors = px.colors.qualitative.Set3
        
    def analyze_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """
        基本統計情報の分析
        
        Args:
            df: 分析するDataFrame
            
        Returns:
            Dict: 基本統計情報
        """
        logger.info("基本統計情報を分析中...")
        
        stats = {
            'dataset_info': {
                'total_reviews': len(df),
                'unique_labels': df['label'].nunique(),
                'label_list': df['label'].unique().tolist(),
            },
            'text_statistics': {
                'avg_length': df['review_text'].str.len().mean(),
                'median_length': df['review_text'].str.len().median(),
                'min_length': df['review_text'].str.len().min(),
                'max_length': df['review_text'].str.len().max(),
                'std_length': df['review_text'].str.len().std(),
            },
            'label_distribution': df['label'].value_counts().to_dict(),
            'label_percentages': (df['label'].value_counts(normalize=True) * 100).to_dict()
        }
        
        # 結果を表示
        print("=== 基本統計情報 ===")
        print(f"総レビュー数: {stats['dataset_info']['total_reviews']:,}")
        print(f"ラベル数: {stats['dataset_info']['unique_labels']}")
        print(f"平均文字数: {stats['text_statistics']['avg_length']:.1f}")
        print(f"中央値文字数: {stats['text_statistics']['median_length']:.1f}")
        
        return stats
    
    def plot_label_distribution(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        ラベル分布の可視化
        
        Args:
            df: 分析するDataFrame
            save_path: 保存パス
        """
        logger.info("ラベル分布を可視化中...")
        
        # matplotlib版
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=EDA_CONFIG['figure_size'])
        
        label_counts = df['label'].value_counts()
        
        # 棒グラフ
        bars = ax1.bar(range(len(label_counts)), label_counts.values, 
                      color=sns.color_palette("husl", len(label_counts)))
        ax1.set_xlabel('ラベル')
        ax1.set_ylabel('件数')
        ax1.set_title('ラベル別レビュー件数')
        ax1.set_xticks(range(len(label_counts)))
        ax1.set_xticklabels(label_counts.index, rotation=45, ha='right')
        
        # 値をバーの上に表示
        for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom')
        
        # 円グラフ
        wedges, texts, autotexts = ax2.pie(label_counts.values, labels=label_counts.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('ラベル分布（割合）')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.figures_dir, 'label_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Plotly版（インタラクティブ）
        fig_plotly = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ラベル別件数', 'ラベル分布'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 棒グラフ
        fig_plotly.add_trace(
            go.Bar(x=label_counts.index, y=label_counts.values,
                   name="件数", marker_color=self.colors[:len(label_counts)]),
            row=1, col=1
        )
        
        # 円グラフ
        fig_plotly.add_trace(
            go.Pie(labels=label_counts.index, values=label_counts.values,
                   name="分布"),
            row=1, col=2
        )
        
        fig_plotly.update_layout(
            title_text="ラベル分布分析",
            height=500,
            showlegend=False
        )
        
        # HTMLとして保存
        plotly_path = os.path.join(self.figures_dir, 'label_distribution_interactive.html')
        fig_plotly.write_html(plotly_path)
        logger.info(f"インタラクティブ図を保存: {plotly_path}")
    
    def plot_text_length_analysis(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        テキスト長の分析
        
        Args:
            df: 分析するDataFrame
            save_path: 保存パス
        """
        logger.info("テキスト長を分析中...")
        
        df['text_length'] = df['review_text'].str.len()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 全体の分布
        axes[0, 0].hist(df['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('文字数')
        axes[0, 0].set_ylabel('頻度')
        axes[0, 0].set_title('テキスト長分布（全体）')
        axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--', 
                          label=f'平均: {df["text_length"].mean():.1f}')
        axes[0, 0].legend()
        
        # ラベル別の分布
        for i, label in enumerate(df['label'].unique()):
            if i >= 10:  # 最大10ラベルまで表示
                break
            subset = df[df['label'] == label]['text_length']
            axes[0, 1].hist(subset, bins=30, alpha=0.5, label=label)
        
        axes[0, 1].set_xlabel('文字数')
        axes[0, 1].set_ylabel('頻度')
        axes[0, 1].set_title('ラベル別テキスト長分布')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # ボックスプロット
        df.boxplot(column='text_length', by='label', ax=axes[1, 0])
        axes[1, 0].set_title('ラベル別テキスト長（ボックスプロット）')
        axes[1, 0].set_xlabel('ラベル')
        axes[1, 0].set_ylabel('文字数')
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # 統計サマリー
        stats_text = f"""
        平均文字数: {df['text_length'].mean():.1f}
        中央値: {df['text_length'].median():.1f}
        標準偏差: {df['text_length'].std():.1f}
        最小値: {df['text_length'].min()}
        最大値: {df['text_length'].max()}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('統計サマリー')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.figures_dir, 'text_length_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_wordcloud_by_label(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """
        ラベル別ワードクラウドの作成
        
        Args:
            df: 分析するDataFrame
            save_dir: 保存ディレクトリ
        """
        logger.info("ラベル別ワードクラウドを作成中...")
        
        if save_dir is None:
            save_dir = self.figures_dir
        
        # 日本語ワードクラウド用の設定
        wordcloud_config = {
            'width': EDA_CONFIG['wordcloud_width'],
            'height': EDA_CONFIG['wordcloud_height'],
            'max_words': EDA_CONFIG['max_wordcloud_words'],
            'background_color': 'white',
            'collocations': False,
            'font_path': None  # システムの日本語フォントを使用
        }
        
        unique_labels = df['label'].unique()
        n_labels = len(unique_labels)
        
        # グリッドサイズを計算
        cols = min(3, n_labels)
        rows = (n_labels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_labels == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, label in enumerate(unique_labels):
            row = i // cols
            col = i % cols
            
            # ラベルのテキストを結合
            label_texts = df[df['label'] == label]['review_text'].tolist()
            combined_text = ' '.join(label_texts)
            
            # 簡単な前処理（記号除去など）
            combined_text = re.sub(r'[!！?？。、,，\[\]()（）]', ' ', combined_text)
            
            try:
                wordcloud = WordCloud(**wordcloud_config).generate(combined_text)
                
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes[0]
                else:
                    ax = axes[row, col]
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'{label}\\n({len(label_texts)}件)', fontsize=12)
                ax.axis('off')
                
                # 個別保存
                individual_path = os.path.join(save_dir, f'wordcloud_{label}.png')
                wordcloud.to_file(individual_path)
                
            except Exception as e:
                logger.warning(f"ワードクラウド作成エラー（{label}）: {e}")
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes[0]
                else:
                    ax = axes[row, col]
                ax.text(0.5, 0.5, f'Error\\n{label}', ha='center', va='center')
                ax.set_title(f'{label}\\n({len(label_texts)}件)')
                ax.axis('off')
        
        # 空のサブプロットを非表示
        for i in range(n_labels, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                ax = axes[col] if cols > 1 else axes[0]
            else:
                ax = axes[row, col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'wordclouds_all_labels.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_class_imbalance(self, df: pd.DataFrame) -> Dict:
        """
        クラス不均衡の分析
        
        Args:
            df: 分析するDataFrame
            
        Returns:
            Dict: 不均衡分析結果
        """
        logger.info("クラス不均衡を分析中...")
        
        label_counts = df['label'].value_counts()
        
        imbalance_analysis = {
            'label_counts': label_counts.to_dict(),
            'imbalance_ratio': label_counts.max() / label_counts.min(),
            'majority_class': label_counts.idxmax(),
            'minority_class': label_counts.idxmin(),
            'majority_percentage': (label_counts.max() / len(df)) * 100,
            'minority_percentage': (label_counts.min() / len(df)) * 100,
        }
        
        # 推奨される対策を提案
        ratio = imbalance_analysis['imbalance_ratio']
        if ratio > 10:
            recommendation = "重度の不均衡：SMOTE、重み付き損失、アンダーサンプリングを検討"
        elif ratio > 5:
            recommendation = "中度の不均衡：クラス重み付けを検討"
        elif ratio > 2:
            recommendation = "軽度の不均衡：層化分割で対応可能"
        else:
            recommendation = "バランス良好：特別な対策不要"
        
        imbalance_analysis['recommendation'] = recommendation
        
        print("\\n=== クラス不均衡分析 ===")
        print(f"不均衡比率: {ratio:.2f}")
        print(f"多数クラス: {imbalance_analysis['majority_class']} ({imbalance_analysis['majority_percentage']:.1f}%)")
        print(f"少数クラス: {imbalance_analysis['minority_class']} ({imbalance_analysis['minority_percentage']:.1f}%)")
        print(f"推奨対策: {recommendation}")
        
        return imbalance_analysis
    
    def show_sample_reviews(self, df: pd.DataFrame, n_samples: int = None) -> None:
        """
        各ラベルのサンプルレビューを表示
        
        Args:
            df: 分析するDataFrame
            n_samples: 表示するサンプル数
        """
        n_samples = n_samples or EDA_CONFIG['sample_size_for_display']
        
        print("\\n=== ラベル別サンプルレビュー ===")
        
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            samples = label_df.sample(min(n_samples, len(label_df)))
            
            print(f"\\n【{label}】({len(label_df)}件):")
            for i, (idx, row) in enumerate(samples.iterrows(), 1):
                text = row['review_text']
                if len(text) > 100:
                    text = text[:100] + "..."
                print(f"  {i}. {text}")
    
    def create_correlation_analysis(self, df: pd.DataFrame) -> None:
        """
        数値特徴量間の相関分析
        
        Args:
            df: 分析するDataFrame
        """
        logger.info("相関分析を実行中...")
        
        # 数値特徴量を作成
        df['text_length'] = df['review_text'].str.len()
        df['word_count'] = df['review_text'].str.split().str.len()
        df['exclamation_count'] = df['review_text'].str.count('[!！]')
        df['question_count'] = df['review_text'].str.count('[?？]')
        
        # ラベルを数値エンコード
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])
        
        # 相関行列
        numeric_features = ['text_length', 'word_count', 'exclamation_count', 
                          'question_count', 'label_encoded']
        corr_matrix = df[numeric_features].corr()
        
        # ヒートマップ
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('特徴量間の相関分析')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correlation_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_eda(self, excel_file_path: str) -> Dict:
        """
        完全なEDA実行
        
        Args:
            excel_file_path: Excelファイルのパス
            
        Returns:
            Dict: EDA結果
        """
        logger.info("=== 完全なEDA分析を開始 ===")
        
        # データ読み込み
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_data(excel_file_path)
        df = processed_data['dataframe']
        
        # 各種分析の実行
        basic_stats = self.analyze_basic_statistics(df)
        
        self.plot_label_distribution(df)
        self.plot_text_length_analysis(df)
        self.create_wordcloud_by_label(df)
        
        imbalance_analysis = self.analyze_class_imbalance(df)
        
        self.show_sample_reviews(df)
        self.create_correlation_analysis(df)
        
        # 結果をまとめる
        eda_results = {
            'basic_statistics': basic_stats,
            'imbalance_analysis': imbalance_analysis,
            'preprocessed_data': processed_data
        }
        
        logger.info("=== EDA分析完了 ===")
        logger.info(f"図は以下に保存されました: {self.figures_dir}")
        
        return eda_results

if __name__ == "__main__":
    # 使用例
    analyzer = EDAAnalyzer()
    
    # サンプルファイルパス（実際のパスに変更してください）
    excel_file = "../ラベル付け参考用.xlsx"
    
    if os.path.exists(excel_file):
        eda_results = analyzer.run_complete_eda(excel_file)
        print("\\nEDA分析が完了しました！")
    else:
        print(f"ファイルが見つかりません: {excel_file}")
