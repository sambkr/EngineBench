import torch


def RI(A, B):
    """
    Compute the cosine similarity between two tensors.

    Parameters:
    - A: PyTorch tensor
    - B: PyTorch tensor

    Returns:
    - Cosine similarity as a float.
    """
    # Flatten the tensors
    A_flat = A.view(-1)
    B_flat = B.view(-1)

    dot_product = torch.dot(A_flat, B_flat)

    norm_A = torch.norm(A_flat)
    norm_B = torch.norm(B_flat)

    # Check for zero magnitude to prevent division by zero
    if norm_A.item() == 0 or norm_B.item() == 0:
        raise ValueError(
            "One of the vectors has zero magnitude, cannot compute cosine similarity."
        )

    cosine_sim = dot_product / (norm_A * norm_B)

    return cosine_sim


def MI(A, B):
    """
    Compute the magnitude index (MI) between two arrays.

    Parameters:
    - A: numpy array.
    - B: numpy array.

    Returns:
    - The magnitude index as a float.
    """
    diff_norm = torch.norm(A - B)

    norm_A = torch.norm(A)
    norm_B = torch.norm(B)

    if norm_A + norm_B == 0:
        raise ValueError(
            "The sum of the norms of both tensors is zero, cannot compute magnitude index."
        )

    mi = 1 - diff_norm / (norm_A + norm_B)

    return mi


def relative_l2(true, predicted):
    """
    Calculate the relative L2 error norm between two tensors in PyTorch.

    Parameters:
    - true: PyTorch tensor of true values.
    - predicted: PyTorch tensor of predicted values.

    Returns:
    - Relative L2 error norm as a float.
    """
    true = true.view(-1)  # Flatten the tensors
    predicted = predicted.view(-1)

    error_norm = torch.norm(true - predicted)
    true_norm = torch.norm(true)

    if true_norm == 0:
        raise ValueError(
            "The L2 norm of the true tensor is zero, cannot compute relative error."
        )

    relative_error = error_norm / true_norm

    return relative_error.item()
