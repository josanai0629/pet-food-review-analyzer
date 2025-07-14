"""
データ前処理モジュール
ペットフードレビューデータの読み込み、清掃、前処理を行う
"""
import pandas as pd
import numpy as np
import re
import os
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATA_CONFIG, PATHS, PREPROCESSING_CONFIG

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """データ前処理クラス"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.data_stats = {}
        
    def load_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        Excelファイルからデータを読み込み
        
        Args:
            file_path: Excelファイルのパス
            
        Returns:
            DataFrame: 読み込んだデータ
        """
        try:
            logger.info(f"Excelファイルを読み込み中: {file_path}")
            df = pd.read_excel(file_path)
            
            # A列：レビューテキスト、B列：ラベルを想定
            if df.shape[1] < 2:
                raise ValueError("データには少なくとも2列（テキスト、ラベル）が必要です")
            
            # 列名を標準化
            df.columns = ['review_text', 'label'] + list(df.columns[2:])
            
            logger.info(f"データ形状: {df.shape}")
            logger.info(f"列名: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        テキストの清掃
        
        Args:
            text: 清掃するテキスト
            
        Returns:
            str: 清掃後のテキスト
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 基本的な正規化
        text = str(text).strip()
        
        if PREPROCESSING_CONFIG['remove_urls']:
            # URL除去
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        if PREPROCESSING_CONFIG['remove_emails']:
            # メールアドレス除去
            text = re.sub(r'\\S+@\\S+', '', text)
        
        if PREPROCESSING_CONFIG['remove_extra_whitespace']:
            # 余分な空白除去
            text = re.sub(r'\\s+', ' ', text)
            text = text.strip()
        
        if PREPROCESSING_CONFIG['normalize_text']:
            # 文字正規化（全角→半角など）
            text = self._normalize_characters(text)
        
        return text
    
    def _normalize_characters(self, text: str) -> str:
        """文字の正規化"""
        # 全角英数字を半角に変換
        text = text.translate(str.maketrans(
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        ))
        
        # 重複する句読点や記号を整理
        text = re.sub(r'[！!]{2,}', '！', text)
        text = re.sub(r'[？?]{2,}', '？', text)
        text = re.sub(r'[。]{2,}', '。', text)
        
        return text
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの検証と清掃
        
        Args:
            df: 検証するDataFrame
            
        Returns:
            DataFrame: 検証・清掃後のデータ
        """
        logger.info("データ検証を開始")
        
        # 元のデータ数を記録
        original_count = len(df)
        
        # ヘッダー行を除去（1行目がヘッダーの場合）
        if df.iloc[0, 0] == 'review_text' or 'review_text' in str(df.iloc[0, 0]).lower():
            df = df.iloc[1:].reset_index(drop=True)
            logger.info("ヘッダー行を除去しました")
        
        # テキストとラベルの清掃
        df['review_text'] = df['review_text'].apply(self.clean_text)
        df['label'] = df['label'].astype(str).str.strip()
        
        # 空のデータを除去
        df = df[
            (df['review_text'].str.len() >= DATA_CONFIG['min_text_length']) &
            (df['review_text'].str.len() <= DATA_CONFIG['max_text_length']) &
            (df['label'] != '') &
            (df['label'] != 'nan') &
            (~df['label'].isna())
        ]
        
        # 重複除去
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['review_text']).reset_index(drop=True)
        after_dedup = len(df)
        
        # 統計情報を保存
        self.data_stats = {
            'original_count': original_count,
            'after_cleaning': len(df),
            'duplicates_removed': before_dedup - after_dedup,
            'text_length_stats': {
                'mean': df['review_text'].str.len().mean(),
                'median': df['review_text'].str.len().median(),
                'min': df['review_text'].str.len().min(),
                'max': df['review_text'].str.len().max(),
            }
        }
        
        logger.info(f"データ検証完了:")
        logger.info(f"  元データ数: {original_count}")
        logger.info(f"  清掃後: {len(df)}")
        logger.info(f"  重複除去: {before_dedup - after_dedup}件")
        logger.info(f"  平均文字数: {self.data_stats['text_length_stats']['mean']:.1f}")
        
        return df
    
    def analyze_labels(self, df: pd.DataFrame) -> Dict:
        """
        ラベルの分析
        
        Args:
            df: 分析するDataFrame
            
        Returns:
            Dict: ラベル分析結果
        """
        label_counts = df['label'].value_counts()
        label_percentages = df['label'].value_counts(normalize=True) * 100
        
        label_analysis = {
            'label_counts': label_counts.to_dict(),
            'label_percentages': label_percentages.to_dict(),
            'num_unique_labels': len(label_counts),
            'most_common_label': label_counts.index[0],
            'least_common_label': label_counts.index[-1],
            'imbalance_ratio': label_counts.iloc[0] / label_counts.iloc[-1]
        }
        
        logger.info("ラベル分析結果:")
        for label, count in label_counts.items():
            percentage = label_percentages[label]
            logger.info(f"  {label}: {count}件 ({percentage:.1f}%)")
        
        return label_analysis
    
    def encode_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        ラベルのエンコード
        
        Args:
            df: エンコードするDataFrame
            
        Returns:
            Tuple: (エンコード済みラベル, 元のラベル)
        """
        original_labels = df['label'].values
        encoded_labels = self.label_encoder.fit_transform(original_labels)
        
        logger.info(f"ラベルエンコード完了: {len(self.label_encoder.classes_)}クラス")
        logger.info(f"クラス一覧: {list(self.label_encoder.classes_)}")
        
        return encoded_labels, original_labels
    
    def split_data(self, texts: List[str], labels: np.ndarray, 
                   test_size: float = None, validation_size: float = None,
                   random_state: int = None) -> Tuple:
        """
        データ分割
        
        Args:
            texts: テキストリスト
            labels: ラベル配列
            test_size: テストデータの割合
            validation_size: 検証データの割合
            random_state: 乱数シード
            
        Returns:
            Tuple: 分割後のデータ
        """
        test_size = test_size or DATA_CONFIG['test_size']
        validation_size = validation_size or DATA_CONFIG['validation_size']
        random_state = random_state or DATA_CONFIG['random_seed']
        
        # まず訓練+検証データとテストデータに分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        # 訓練データと検証データに分割
        if validation_size > 0:
            val_size = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=random_state,
                stratify=y_temp
            )
            
            logger.info(f"データ分割完了:")
            logger.info(f"  訓練: {len(X_train)}件")
            logger.info(f"  検証: {len(X_val)}件")
            logger.info(f"  テスト: {len(X_test)}件")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            logger.info(f"データ分割完了:")
            logger.info(f"  訓練: {len(X_temp)}件")
            logger.info(f"  テスト: {len(X_test)}件")
            
            return X_temp, X_test, y_temp, y_test
    
    def save_processed_data(self, processed_data: Dict, output_path: str = None):
        """
        前処理済みデータの保存
        
        Args:
            processed_data: 前処理済みデータ
            output_path: 出力パス
        """
        if output_path is None:
            os.makedirs(PATHS['data_dir'], exist_ok=True)
            output_path = os.path.join(PATHS['data_dir'], 'processed_data.pkl')
        
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"前処理済みデータを保存: {output_path}")
    
    def process_data(self, excel_file_path: str) -> Dict:
        """
        データ前処理のメインフロー
        
        Args:
            excel_file_path: Excelファイルのパス
            
        Returns:
            Dict: 前処理済みデータ
        """
        logger.info("=== データ前処理開始 ===")
        
        # データ読み込み
        df = self.load_excel_data(excel_file_path)
        
        # データ検証・清掃
        df = self.validate_data(df)
        
        # ラベル分析
        label_analysis = self.analyze_labels(df)
        
        # ラベルエンコード
        encoded_labels, original_labels = self.encode_labels(df)
        
        # テキスト抽出
        texts = df['review_text'].tolist()
        
        # データ分割
        split_result = self.split_data(texts, encoded_labels)
        
        # 結果をまとめる
        processed_data = {
            'texts': texts,
            'labels': encoded_labels,
            'original_labels': original_labels,
            'label_encoder': self.label_encoder,
            'label_analysis': label_analysis,
            'data_stats': self.data_stats,
            'split_data': split_result,
            'dataframe': df
        }
        
        logger.info("=== データ前処理完了 ===")
        
        return processed_data

if __name__ == "__main__":
    # 使用例
    preprocessor = DataPreprocessor()
    
    # サンプルファイルパス（実際のパスに変更してください）
    excel_file = "../ラベル付け参考用.xlsx"
    
    if os.path.exists(excel_file):
        processed_data = preprocessor.process_data(excel_file)
        
        # 結果の保存
        preprocessor.save_processed_data(processed_data)
        
        print("\\n=== 前処理結果サマリー ===")
        print(f"総データ数: {len(processed_data['texts'])}")
        print(f"ラベル数: {len(preprocessor.label_encoder.classes_)}")
        print(f"ラベル一覧: {list(preprocessor.label_encoder.classes_)}")
    else:
        print(f"ファイルが見つかりません: {excel_file}")
