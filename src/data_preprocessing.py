"""
Data Preprocessing Module
Handles data loading, cleaning, and train/dev/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import os


def load_dataset(filepath):
    """
    Load the email spam dataset
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    return df


def combine_text_features(df):
    """
    Combine title and text columns into a single text field
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with combined text
    """
    df = df.copy()
    # Handle NaN values
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Combine title and text
    df['combined_text'] = df['title'] + ' ' + df['text']
    df['combined_text'] = df['combined_text'].str.strip()
    
    return df


def clean_text(text, lowercase=True, remove_urls=True, remove_extra_spaces=True):
    """
    Clean and preprocess text
    
    Args:
        text (str): Input text
        lowercase (bool): Convert to lowercase
        remove_urls (bool): Remove URLs
        remove_extra_spaces (bool): Remove extra whitespace
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    text = text.strip()
    
    return text


def preprocess_dataset(df, clean=True):
    """
    Preprocess the entire dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        clean (bool): Whether to apply text cleaning
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.copy()
    
    # Combine text features
    df = combine_text_features(df)
    
    # Clean text
    if clean:
        df['cleaned_text'] = df['combined_text'].apply(clean_text)
    else:
        df['cleaned_text'] = df['combined_text']
    
    # Convert labels to binary (1 for spam, 0 for not spam)
    df['label'] = (df['type'] == 'spam').astype(int)
    
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nSpam percentage: {df['label'].mean() * 100:.2f}%")
    
    return df


def split_dataset(input_path='data/email_spam.csv', 
                  train_size=0.70, 
                  dev_size=0.15, 
                  test_size=0.15, 
                  random_state=42,
                  output_dir='data'):
    """
    Split dataset into train, dev, and test sets with stratification
    
    Args:
        input_path (str): Path to input CSV file
        train_size (float): Proportion for training set
        dev_size (float): Proportion for development set
        test_size (float): Proportion for test set
        random_state (int): Random seed for reproducibility
        output_dir (str): Directory to save split datasets
        
    Returns:
        tuple: (train_df, dev_df, test_df)
    """
    # Validate split proportions
    assert abs(train_size + dev_size + test_size - 1.0) < 1e-6, \
        "Split proportions must sum to 1.0"
    
    # Load and preprocess data
    df = load_dataset(input_path)
    df = preprocess_dataset(df)
    
    # Remove any rows with empty text
    df = df[df['cleaned_text'].str.strip() != '']
    print(f"After removing empty texts: {len(df)} samples")
    
    # First split: separate test set
    train_dev_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Second split: separate dev from train
    # Adjust dev_size relative to remaining data
    relative_dev_size = dev_size / (train_size + dev_size)
    train_df, dev_df = train_test_split(
        train_dev_df,
        test_size=relative_dev_size,
        random_state=random_state,
        stratify=train_dev_df['label']
    )
    
    print(f"\n{'='*50}")
    print(f"Dataset Split Summary:")
    print(f"{'='*50}")
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Spam: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    print(f"  - Ham: {len(train_df) - train_df['label'].sum()} ({(1-train_df['label'].mean())*100:.1f}%)")
    
    print(f"\nDevelopment set: {len(dev_df)} samples ({len(dev_df)/len(df)*100:.1f}%)")
    print(f"  - Spam: {dev_df['label'].sum()} ({dev_df['label'].mean()*100:.1f}%)")
    print(f"  - Ham: {len(dev_df) - dev_df['label'].sum()} ({(1-dev_df['label'].mean())*100:.1f}%)")
    
    print(f"\nTest set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  - Spam: {test_df['label'].sum()} ({test_df['label'].mean()*100:.1f}%)")
    print(f"  - Ham: {len(test_df) - test_df['label'].sum()} ({(1-test_df['label'].mean())*100:.1f}%)")
    print(f"{'='*50}\n")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(output_dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Splits saved to {output_dir}/")
    
    return train_df, dev_df, test_df


def load_split_data(data_dir='data'):
    """
    Load pre-split train, dev, and test datasets
    
    Args:
        data_dir (str): Directory containing split datasets
        
    Returns:
        tuple: (train_df, dev_df, test_df)
    """
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(data_dir, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print(f"Loaded train: {len(train_df)}, dev: {len(dev_df)}, test: {len(test_df)}")
    
    return train_df, dev_df, test_df


def get_text_and_labels(df):
    """
    Extract text and labels from dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (texts, labels)
    """
    texts = df['cleaned_text'].values
    labels = df['label'].values
    
    return texts, labels


if __name__ == "__main__":
    # Example usage
    print("Splitting dataset...")
    train_df, dev_df, test_df = split_dataset()
    
    print("\nDataset split complete!")