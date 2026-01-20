import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture

def extract_latent(encoder, dataloader, classifier=None, device='cuda'):
    encoder.eval()
    if classifier is not None:
        classifier.eval()

    latents = []
    labels = []
    confidences = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # Get latent features
            z = encoder(x)
            latents.append(z.cpu().numpy())
            labels.append(y.cpu().numpy())

            # Compute confidence if classifier provided
            if classifier is not None:
                logits = classifier(z)
                probs = F.softmax(logits, dim=1)
                conf, _ = torch.max(probs, dim=1)  # confidence = max softmax prob
                confidences.append(conf.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    confidences = np.concatenate(confidences, axis=0) if confidences else None

    return latents, labels, confidences
def detector(source_extractor, classifier, target_loader, device, threshold_percentile=1, conf_threshold=0.7):
    """
    Detect anomalies in the target domain using a GMM trained on high-confidence normal samples.

    Args:
        source_extractor (torch.nn.Module): Feature extractor model.
        classifier (torch.nn.Module): Classifier model.
        target_loader (DataLoader): Dataloader for the target dataset.
        device (torch.device): Device to run inference on ('cpu' or 'cuda').
        threshold_percentile (float): Percentile for anomaly threshold (default=1).
        conf_threshold (float): Confidence threshold for pseudo-normal selection (default=0.7).

    Returns:
        pred_gmm (np.ndarray): Predicted anomaly labels (0=normal, 1=attack).
        ground_truth (np.ndarray): True labels.
        labels (np.ndarray): Classifier predicted labels.
        confidences (np.ndarray): Prediction confidences.
        features (np.ndarray): Extracted feature embeddings.
    """

    features, labels, confidences, ground_truth = [], [], [], []

    source_extractor.eval()
    classifier.eval()

    with torch.no_grad():
        for x, y in target_loader:
            x = x.to(device)
            z = source_extractor(x)
            logits = classifier(z)
            probs = torch.softmax(logits, dim=1)

            conf, pred = torch.max(probs, dim=1)

            ground_truth.append(y.cpu().numpy())
            features.append(z.cpu().numpy())
            labels.append(pred.cpu().numpy())
            confidences.append(conf.cpu().numpy())

    # Combine all batches
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    ground_truth = np.concatenate(ground_truth)
    confidences = np.concatenate(confidences)

    # Select confident normal samples
    mask_normal = (labels == 0) & (confidences > conf_threshold)
    X_pseudo_normal = features[mask_normal]

    # Fit GMM on pseudo-normal samples
    gmm_normal = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm_normal.fit(X_pseudo_normal)

    # Compute normality scores and threshold
    normal_scores = gmm_normal.score_samples(X_pseudo_normal)
    threshold = np.percentile(normal_scores, threshold_percentile)

    # Score all samples
    feature_scores = gmm_normal.score_samples(features)

    # Predict anomalies
    pred_gmm = (feature_scores <= threshold).astype(int)

    return pred_gmm ,ground_truth