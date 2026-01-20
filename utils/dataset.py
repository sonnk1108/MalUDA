import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
class TabularImageDataset(Dataset):
    def __init__(self, c1, y, img_height, img_width):
        """
        c1: array/list of shape (num_samples, feature_length) for the single channel
        y: labels
        img_height, img_width: target height and width of the image
        """
        assert len(c1) == len(y), "Input and labels must have the same number of samples"
        
        self.y = torch.tensor(y, dtype=torch.long)
        self.X_img = []
        
        # Determine the padded length to match the image size
        padded_len = img_height * img_width
        
        for i in range(len(y)):
            # Pad the feature vector to match the image size
            row_padded = np.pad(c1[i], (0, padded_len - len(c1[i])), 'constant')
            # Reshape to 2D image
            img = row_padded.reshape(img_height, img_width)
            # Add channel dimension -> shape: (1, H, W)
            img_1ch = img[np.newaxis, :, :]
            self.X_img.append(img_1ch)
        
        self.X_img = torch.tensor(np.array(self.X_img), dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_img[idx], self.y[idx]


class TabularDataset(Dataset):
    def __init__(self, X, y, dtype=torch.float32):
        """
        X : numpy array or pandas DataFrame of shape (N, F)
        y : labels
        dtype : data type for features
        """
        # Convert dataframe to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        assert len(X) == len(y), "X and y must have the same number of samples"

        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=torch.long)  # classification labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
