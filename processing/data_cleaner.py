"""Data cleaning and normalization module."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import re
import logging
from PIL import Image
import io
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and normalize heterogeneous data."""
    
    def __init__(self):
        self.cleaned_data = {}
    
    def clean_text(self, text: str) -> str:
        """Clean text data by removing extra whitespace and special characters."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"()-]', '', text)
        
        return text
    
    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize table data."""
        if df.empty:
            return df
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # For text columns, fill with empty string
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            df[col].fillna('', inplace=True)
        
        # Normalize column names
        df.columns = [re.sub(r'\W+', '_', col.lower().strip()) for col in df.columns]
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        logger.info(f"Cleaned table with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def normalize_images(self, image_paths: List[str], 
                        target_size: tuple = (224, 224),
                        output_dir: str = "data/processed/images") -> List[str]:
        """Normalize images to consistent size and format."""
        os.makedirs(output_dir, exist_ok=True)
        normalized_paths = []
        
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize image
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Save normalized image
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(output_dir, filename)
                    img.save(output_path, 'JPEG', quality=95)
                    normalized_paths.append(output_path)
                    
                    logger.info(f"Normalized image: {filename}")
            except Exception as e:
                logger.error(f"Error normalizing image {img_path}: {e}")
        
        return normalized_paths
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        df_clean = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                                   (df_clean[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_clean = df_clean[z_scores < threshold]
        
        logger.info(f"Removed {len(df) - len(df_clean)} outliers")
        return df_clean
    
    def standardize_text_encoding(self, text: str) -> str:
        """Standardize text encoding to UTF-8."""
        if isinstance(text, bytes):
            return text.decode('utf-8', errors='ignore')
        return text
    
    def batch_clean_tables(self, table_dir: str) -> List[pd.DataFrame]:
        """Clean all CSV tables in a directory."""
        cleaned_tables = []
        
        for filename in os.listdir(table_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(table_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    df_clean = self.clean_table(df)
                    cleaned_tables.append(df_clean)
                    
                    # Save cleaned table
                    output_path = filepath.replace('/raw/', '/processed/')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    df_clean.to_csv(output_path, index=False)
                except Exception as e:
                    logger.error(f"Error cleaning table {filename}: {e}")
        
        return cleaned_tables

if __name__ == "__main__":
    cleaner = DataCleaner()
    print("Data cleaner module loaded successfully")
