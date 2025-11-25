"""
BERT Model Module
Implements DistilBERT fine-tuning for spam classification
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
import os


class EmailDataset(Dataset):
    """
    PyTorch Dataset for email texts
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset
        
        Args:
            texts (list): List of email texts
            labels (array): Array of labels
            tokenizer: BERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTSpamClassifier:
    """
    BERT-based spam classifier
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, 
                 max_length=512, device=None):
        """
        Initialize BERT classifier
        
        Args:
            model_name (str): Pre-trained model name
            num_labels (int): Number of output labels
            max_length (int): Maximum sequence length
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
    def create_data_loader(self, texts, labels, batch_size=16, shuffle=True):
        """
        Create PyTorch DataLoader
        
        Args:
            texts (list): List of texts
            labels (array): Array of labels
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data
            
        Returns:
            DataLoader
        """
        dataset = EmailDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_epoch(self, data_loader, optimizer, scheduler):
        """
        Train for one epoch
        
        Args:
            data_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(data_loader, desc='Training')
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
        """
        Evaluate the model
        
        Args:
            data_loader: Evaluation data loader
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        
        predictions = []
        true_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        accuracy = np.mean(predictions == true_labels)
        
        # Calculate precision, recall, F1 for spam class (label 1)
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'true_labels': true_labels,
            'confusion_matrix': np.array([[tn, fp], [fn, tp]])
        }
        
        return metrics
    
    def train(self, train_texts, train_labels, dev_texts, dev_labels,
              epochs=5, batch_size=16, learning_rate=2e-5, warmup_steps=100):
        """
        Train the model
        
        Args:
            train_texts (list): Training texts
            train_labels (array): Training labels
            dev_texts (list): Development texts
            dev_labels (array): Development labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            warmup_steps (int): Warmup steps for scheduler
            
        Returns:
            dict: Training history
        """
        # Create data loaders
        train_loader = self.create_data_loader(train_texts, train_labels, 
                                               batch_size, shuffle=True)
        dev_loader = self.create_data_loader(dev_texts, dev_labels, 
                                             batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {
            'train_loss': [],
            'dev_loss': [],
            'dev_accuracy': [],
            'dev_f1': []
        }
        
        best_f1 = 0
        
        print(f"\n{'='*60}")
        print("STARTING BERT TRAINING")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(dev_texts)}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            print(f"Average training loss: {train_loss:.4f}")
            
            # Evaluate
            dev_metrics = self.evaluate(dev_loader)
            
            print(f"\nDevelopment Set Metrics:")
            print(f"  Loss: {dev_metrics['loss']:.4f}")
            print(f"  Accuracy: {dev_metrics['accuracy']:.4f}")
            print(f"  Precision: {dev_metrics['precision']:.4f}")
            print(f"  Recall: {dev_metrics['recall']:.4f}")
            print(f"  F1 Score: {dev_metrics['f1_score']:.4f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['dev_loss'].append(dev_metrics['loss'])
            history['dev_accuracy'].append(dev_metrics['accuracy'])
            history['dev_f1'].append(dev_metrics['f1_score'])
            
            # Save best model
            if dev_metrics['f1_score'] > best_f1:
                best_f1 = dev_metrics['f1_score']
                print(f"\n✓ New best F1 score: {best_f1:.4f}")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Best Dev F1 Score: {best_f1:.4f}")
        print(f"{'='*60}\n")
        
        return history
    
    def predict(self, texts, batch_size=16):
        """
        Make predictions on texts
        
        Args:
            texts (list): List of texts
            batch_size (int): Batch size
            
        Returns:
            array: Predictions
        """
        # Create dummy labels (won't be used)
        dummy_labels = np.zeros(len(texts))
        data_loader = self.create_data_loader(texts, dummy_labels, 
                                              batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def save(self, save_dir):
        """
        Save model and tokenizer
        
        Args:
            save_dir (str): Directory to save the model
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
    
    def load(self, load_dir):
        """
        Load model and tokenizer
        
        Args:
            load_dir (str): Directory to load the model from
        """
        self.model = DistilBertForSequenceClassification.from_pretrained(load_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(load_dir)
        self.model.to(self.device)
        print(f"Model loaded from {load_dir}")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_split_data, get_text_and_labels
    
    print("Loading data...")
    train_df, dev_df, test_df = load_split_data()
    
    train_texts, train_labels = get_text_and_labels(train_df)
    dev_texts, dev_labels = get_text_and_labels(dev_df)
    test_texts, test_labels = get_text_and_labels(test_df)
    
    print("\nInitializing BERT classifier...")
    classifier = BERTSpamClassifier()
    
    print("\nTraining BERT...")
    history = classifier.train(
        train_texts, train_labels,
        dev_texts, dev_labels,
        epochs=3,
        batch_size=8
    )
    
    print("\nEvaluating on test set...")
    test_loader = classifier.create_data_loader(test_texts, test_labels, 
                                                batch_size=8, shuffle=False)
    test_metrics = classifier.evaluate(test_loader)
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")