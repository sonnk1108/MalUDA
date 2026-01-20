import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def evaluate_target_and_source(target_data_loader, target_extractor, classifier, 
                               source_data_loader, source_extractor, average='binary', 
                               device='cuda'):
    """
    Evaluate target performance and source classification error.

    Args:
        target_data_loader (DataLoader): DataLoader for target domain (with labels)
        target_extractor (nn.Module): Feature extractor for target domain
        classifier (nn.Module): Trained classifier
        source_data_loader (DataLoader): DataLoader for source domain (with labels)
        source_extractor (nn.Module): Feature extractor for source domain
        average (str): 'binary', 'macro', or 'weighted' for multi-class metrics
        device (str): 'cuda' or 'cpu'

    Returns:
        (target_acc, target_prec, target_rec, target_f1, source_error)
    """
    target_extractor.eval()
    source_extractor.eval()
    classifier.eval()
    
    # --- Target performance ---
    all_target_preds = []
    all_target_labels = []
    
    with torch.no_grad():
        for data, labels in target_data_loader:
            data, labels = data.to(device), labels.to(device)
            features = target_extractor(data)
            logits = classifier(features)
            preds = torch.argmax(logits, dim=1)
            all_target_preds.extend(preds.cpu().numpy())
            all_target_labels.extend(labels.cpu().numpy())
    
    target_acc = accuracy_score(all_target_labels, all_target_preds)
    target_prec = precision_score(all_target_labels, all_target_preds, average=average, zero_division=0)
    target_rec = recall_score(all_target_labels, all_target_preds, average=average, zero_division=0)
    target_f1 = f1_score(all_target_labels, all_target_preds, average=average, zero_division=0)
    
    # --- Source classification error ---
    correct_source = 0
    total_source = 0
    
    with torch.no_grad():
        for data, labels in source_data_loader:
            data, labels = data.to(device), labels.to(device)
            features = source_extractor(data)
            logits = classifier(features)
            preds = torch.argmax(logits, dim=1)
            correct_source += torch.sum(preds == labels).item()
            total_source += labels.size(0)
    
    source_error = 1 - (correct_source / total_source)
    
    return target_acc, target_prec, target_rec, target_f1, source_error

def evaluate_target_performance(target_data_loader, target_extractor, classifier, has_labels=False, average='binary', device='cuda'):
    """
    Evaluate classifier performance on the target domain.

    Args:
        target_data_loader (DataLoader): DataLoader for target domain
        target_extractor (nn.Module): Trained feature extractor
        classifier (nn.Module): Trained classifier
        has_labels (bool): Whether target_loader returns (data, label)
        average (str): 'binary', 'macro', or 'weighted' (for multi-class)
        device (str): Device to run inference on ('cuda' or 'cpu')

    Returns:
        (acc, prec, rec, f1): Performance metrics
    """
    target_extractor.eval()
    classifier.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in target_data_loader:
            if has_labels:
                data, labels = batch
                all_labels.extend(labels.cpu().numpy())
            else:
                data = batch[0]

            data = data.to(device)

            # Forward through feature extractor + classifier
            target_features = target_extractor(data)
            logits = classifier(target_features)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # Only compute metrics if labels exist
    if has_labels:
        acc = accuracy_score(all_labels, all_predictions)
        prec = precision_score(all_labels, all_predictions, average=average, zero_division=0)
        rec = recall_score(all_labels, all_predictions, average=average, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average=average, zero_division=0)
        return acc, prec, rec, f1
    else:
        print("[Warning] No ground-truth labels provided. Returning predictions only.")
        return np.array(all_predictions)
