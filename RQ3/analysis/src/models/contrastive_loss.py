import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        # Normalize to unit sphere
        features = F.normalize(features, dim=1)
        
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        # Mask to identify samples with the same label
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute dot products (similarities)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast (the diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        return -mean_log_prob_pos.mean()