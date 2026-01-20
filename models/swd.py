import torch
import torch.nn.functional as F

def swd_loss(source_feature, target_feature, dir_repeats=2, dirs_per_repeat=512, device='cpu'):
    # Check that feature dimensions (columns) match, even if batch sizes (rows) don't
    if source_feature.shape[1] != target_feature.shape[1]:
        raise ValueError("Feature dimensions must match between source and target.")

    feature_dim = source_feature.shape[1]
    n_s = source_feature.shape[0]
    n_t = target_feature.shape[0]
    total_loss = 0

    for _ in range(dir_repeats):
        # Generate and normalize random projections
        theta = torch.randn(feature_dim, dirs_per_repeat, device=device)
        theta = theta / torch.norm(theta, dim=0, keepdim=True)

        # Project features
        proj_s = source_feature.matmul(theta) # [n_s, dirs]
        proj_t = target_feature.matmul(theta) # [n_t, dirs]

        # Sort each projection to get the 1D Empirical CDF
        proj_s, _ = torch.sort(proj_s, dim=0)
        proj_t, _ = torch.sort(proj_t, dim=0)

        # QUANTILE MATCHING: If batch sizes differ, interpolate to the larger size
        if n_s != n_t:
            # Reshape to [Channels, 1, Length] for torch interpolate
            # We treat 'dirs' as channels and the samples as the spatial dimension
            t_proj_s = proj_s.t().unsqueeze(1) 
            t_proj_t = proj_t.t().unsqueeze(1)
            
            target_size = max(n_s, n_t)
            
            t_proj_s = F.interpolate(t_proj_s, size=target_size, mode='linear', align_corners=True)
            t_proj_t = F.interpolate(t_proj_t, size=target_size, mode='linear', align_corners=True)
            
            # Reshape back to [target_size, dirs]
            proj_s = t_proj_s.squeeze(1).t()
            proj_t = t_proj_t.squeeze(1).t()

        # Compute the W1 distance (L1 norm)
        total_loss += torch.abs(proj_s - proj_t).mean()

    return total_loss / dir_repeats