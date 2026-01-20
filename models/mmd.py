import torch
import torch.nn as nn

# --- 2. MMD Loss Function ---
# This function calculates the Maximum Mean Discrepancy between two sets of features.
# A Gaussian kernel (Radial Basis Function) is a common and effective choice.
def mmd_loss(x, y, kernel_type='gaussian', kernel_mul=2.0, kernel_num=5):
    """
    Calculates the MMD loss between two batches of features.
    
    Args:
        x (torch.Tensor): Features from domain A.
        y (torch.Tensor): Features from domain B.
        kernel_type (str): The type of kernel to use (e.g., 'gaussian').
        kernel_mul (float): A scaling factor for the kernel bandwidth.
        kernel_num (int): The number of kernels to use for multi-kernel MMD.
        
    Returns:
        torch.Tensor: The MMD loss value.
    """
    def gaussian_kernel(x_i, y_j, gamma):
        """Helper function for the Gaussian kernel."""
        # Calculate the pairwise squared Euclidean distances
        x_i = x_i.unsqueeze(1)
        y_j = y_j.unsqueeze(0)
        dist = ((x_i - y_j)**2).sum(dim=-1)
        # Apply the RBF kernel
        return torch.exp(-dist / gamma)

    def get_gamma_list(x_i, y_j, kernel_num, kernel_mul):
        """
        Generates a list of gamma values for multi-kernel MMD.
        """
        dist_sq = torch.cdist(x_i, y_j, p=2.0)**2
        median_dist = torch.median(dist_sq.detach()).item()
        
        # Heuristic to set the range of gamma values
        gamma_list = [median_dist / (kernel_mul**i) for i in range(kernel_num)]
        return gamma_list

    # Get the kernel matrix for both domains and cross-domain
    x_x_kernel = 0
    y_y_kernel = 0
    x_y_kernel = 0

    gammas = get_gamma_list(x, y, kernel_num, kernel_mul)

    for gamma in gammas:
        x_x_kernel += gaussian_kernel(x, x, gamma)
        y_y_kernel += gaussian_kernel(y, y, gamma)
        x_y_kernel += gaussian_kernel(x, y, gamma)

    # Calculate the unbiased MMD estimate
    loss = torch.mean(x_x_kernel) + torch.mean(y_y_kernel) - 2 * torch.mean(x_y_kernel)
    return loss

# --- 3. Model Architecture ---
# The core idea of HDA is to use different feature extractors for each domain
# to map them to a single, shared latent space.

