import torch
import torch.nn as nn
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# ------------------------------
# Residual Block (no change)
# ------------------------------
import torch
import torch.nn as nn

class SourceNExtractor(nn.Module):
    """
    Deep MLP feature extractor for TABULAR data (50 features)
    No residual connections
    """
    def __init__(self, input_dim=50, latent_dim=64):
        super(SourceNExtractor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.network(x)

class SourceEExtractor(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(SourceEExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Final layer: NO BatchNorm, NO ReLU
            nn.Linear(128, latent_dim),
        )
    
    def forward(self, x):
        return self.network(x)

# ------------------------------
# Source Feature Extractor
# ------------------------------
class SourceExtractor(nn.Module):
    """
    Pretrained ResNet18 feature extractor with output dimension = latent_dim
    """
    def __init__(self, latent_dim=64):
        super(SourceExtractor, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])  # output: [B, 512, 1, 1]
        # New FC layer to get latent_dim features
        self.fc = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
import torch
import torch.nn as nn

class FriendExtractor(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(FriendExtractor, self).__init__()
        
        # Dense(50, activation="relu") x 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim),
            nn.ReLU()
        )
        self._init_weights()

    def _init_weights(self):
        # Matching kernel_initializer=GlorotUniform(seed=0)
        torch.manual_seed(0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

class FriendClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=2):
        super(FriendClassifier, self).__init__()
        # Dense(1, activation="sigmoid") translated for CrossEntropy
        self.fc = nn.Linear(latent_dim, num_classes)
        
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)