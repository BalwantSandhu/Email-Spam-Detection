"""
Feature Extraction Module
Handles TF-IDF and other feature engineering
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pickle
import os


class FeatureExtractor:
    """
    Feature extraction class for text data
    """
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2), 
                 min_df=2, max_df=0.95, use_idf=True, sublinear_tf=True):
        """
        Initialize feature extractor
        
        Args:
            method (str): Feature extraction method ('tfidf' or 'count')
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to consider
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            use_idf (bool): Enable IDF weighting
            sublinear_tf (bool): Apply sublinear TF scaling
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                use_idf=use_idf,
                sublinear_tf=sublinear_tf,
                strip_accents='unicode',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
            )
        elif method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                strip_accents='unicode',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b'
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, texts):
        """
        Fit the vectorizer on training texts
        
        Args:
            texts (list): List of text documents
            
        Returns:
            self
        """
        print(f"Fitting {self.method} vectorizer on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self
    
    def transform(self, texts):
        """
        Transform texts to feature vectors
        
        Args:
            texts (list): List of text documents
            
        Returns:
            sparse matrix: Feature vectors
        """
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        Fit and transform texts
        
        Args:
            texts (list): List of text documents
            
        Returns:
            sparse matrix: Feature vectors
        """
        print(f"Fitting and transforming {len(texts)} documents...")
        features = self.vectorizer.fit_transform(texts)
        print(f"Feature matrix shape: {features.shape}")
        return features
    
    def get_feature_names(self):
        """
        Get feature names (words/n-grams)
        
        Returns:
            list: Feature names
        """
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=20):
        """
        Get top features by average TF-IDF score
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top feature names
        """
        feature_names = self.get_feature_names()
        return feature_names[:n] if len(feature_names) >= n else feature_names
    
    def save(self, filepath):
        """
        Save the vectorizer to disk
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the vectorizer from disk
        
        Args:
            filepath (str): Path to load the vectorizer from
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"Vectorizer loaded from {filepath}")
        return self


def extract_tfidf_features(train_texts, dev_texts, test_texts, 
                           max_features=5000, ngram_range=(1, 2),
                           min_df=2, max_df=0.95):
    """
    Extract TF-IDF features from train, dev, and test sets
    
    Args:
        train_texts (list): Training texts
        dev_texts (list): Development texts
        test_texts (list): Test texts
        max_features (int): Maximum number of features
        ngram_range (tuple): N-gram range
        min_df (int): Minimum document frequency
        max_df (float): Maximum document frequency
        
    Returns:
        tuple: (train_features, dev_features, test_features, feature_extractor)
    """
    extractor = FeatureExtractor(
        method='tfidf',
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    
    # Fit on training data only
    train_features = extractor.fit_transform(train_texts)
    
    # Transform dev and test data
    dev_features = extractor.transform(dev_texts)
    test_features = extractor.transform(test_texts)
    
    print(f"\nFeature Extraction Complete:")
    print(f"Train features shape: {train_features.shape}")
    print(f"Dev features shape: {dev_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    return train_features, dev_features, test_features, extractor


def analyze_features(feature_extractor, X, y, top_n=20):
    """
    Analyze and display top features for each class
    
    Args:
        feature_extractor (FeatureExtractor): Fitted feature extractor
        X (sparse matrix): Feature matrix
        y (array): Labels
        top_n (int): Number of top features to display
    """
    feature_names = feature_extractor.get_feature_names()
    
    # Calculate mean TF-IDF scores for each class
    spam_indices = np.where(y == 1)[0]
    ham_indices = np.where(y == 0)[0]
    
    if len(spam_indices) > 0:
        spam_scores = np.asarray(X[spam_indices].mean(axis=0)).flatten()
        top_spam_indices = spam_scores.argsort()[-top_n:][::-1]
        
        print(f"\nTop {top_n} features for SPAM:")
        for idx in top_spam_indices:
            print(f"  {feature_names[idx]}: {spam_scores[idx]:.4f}")
    
    if len(ham_indices) > 0:
        ham_scores = np.asarray(X[ham_indices].mean(axis=0)).flatten()
        top_ham_indices = ham_scores.argsort()[-top_n:][::-1]
        
        print(f"\nTop {top_n} features for HAM:")
        for idx in top_ham_indices:
            print(f"  {feature_names[idx]}: {ham_scores[idx]:.4f}")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_split_data, get_text_and_labels
    
    print("Loading data...")
    train_df, dev_df, test_df = load_split_data()
    
    train_texts, train_labels = get_text_and_labels(train_df)
    dev_texts, dev_labels = get_text_and_labels(dev_df)
    test_texts, test_labels = get_text_and_labels(test_df)
    
    print("\nExtracting TF-IDF features...")
    train_X, dev_X, test_X, extractor = extract_tfidf_features(
        train_texts, dev_texts, test_texts
    )
    
    print("\nAnalyzing features...")
    analyze_features(extractor, train_X, train_labels)