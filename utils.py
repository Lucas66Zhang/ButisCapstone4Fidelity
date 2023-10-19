import torch
from torch.utils.data import Dataset
import torch.nn as nn

class Dataset4Dependency(Dataset):
    # Dataset for dependency parsing
    def __init__(self, embeddings: torch.tensor, weights: torch.tensor, labels: torch.tensor):
        """
        init function for Dataset4Dependency
        Args:
            embeddings: dependency embeddings
            weights: dependency weights, {1, 0.8, 0.64}
            labels: dependency labels, 0 for negative, 1 for positive
        """
        # check the number of samples
        assert embeddings.shape[0] == weights.shape[0] == labels.shape[0]
        # check the weights of dependencies
        assert torch.all((weights == 1) | (weights == 0.8) | (weights == 0.64))
        # check the label of dependencies
        assert torch.all((labels == 0) | (labels == 1))

        self.embeddings = embeddings
        self.labels = labels
        self.weights = weights


    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.weights, self.labels[idx]

class CustomizeNLLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CustomizeNLLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.weight = weight
        assert reduction in ['mean', 'sum']
        self.reduction = reduction

    def forward(self, input, target, weights):
        """
        calculate the NLLLoss with sample-level weights
        Args:
            input: prediction from a model
            target: ground truth
            weights: sample-level weights

        Returns:
            loss: weighted NLLLoss
        """
        loss = self.nll_loss(input, target)
        loss = loss * weights
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
