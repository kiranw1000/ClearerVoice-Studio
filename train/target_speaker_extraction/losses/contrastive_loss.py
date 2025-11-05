import torch
import torch.nn as nn
from typing import Annotated

def contrastive_loss(representations: Annotated[torch.Tensor, "Batch * 2 x Features x Time steps"],
                     positive: Annotated[torch.Tensor, "Batch"],
                     temperature: float = 0.5) -> torch.Tensor:
    '''
    Computes the contrastive loss between the representations and samples.
    Args:
        representations (torch.Tensor): Tensor of shape (B * 2, F, T) representing the model outputs. First half are anchor samples, second half are positive samples.
        positive (torch.Tensor): Tensor of shape (B) representing whether the samples are positive.
        temperature (float): Temperature parameter for scaling the logits.
    Returns:
        torch.Tensor: Tensor of shape(B, T) representing the contrastive loss for each time step.
    '''
    assert representations.dim() == 3, f"Representations must be a 3D tensor. Was shape {representations.shape}"
    assert positive.dim() == 1, f"Positive samples must be a 1D tensor. Was shape {positive.shape}"
    B, F, T = representations.shape[0] // 2, representations.shape[1], representations.shape[2]
    assert representations.shape[0] == 2 * B, f"First dimension of representations ({representations.shape[0]}) must be twice the batch size ({2 * B})"
    assert positive.shape[0] == B, f"First dimension of positive ({positive.shape[0]}) must match batch size ({B})"
    ret = torch.zeros(B, T).to(representations.device)
    cos_sim = nn.CosineSimilarity(dim=1)
    for i in range(B):
        s1 = representations[i]
        s2 = representations[i + B]
        logits = cos_sim(s1.unsqueeze(0), s2.unsqueeze(0))
        if positive[i]:
            logits = 1 - logits  # Positive pair
        ret[i] = logits / temperature
    return ret

def triplet_loss(representations: Annotated[torch.Tensor, "Batch * 3 x Features x Time steps"],
                 margin: float = 1.0) -> torch.Tensor:
    '''
    Computes the triplet loss between the representations and samples.
    Args:
        representations (torch.Tensor): Tensor of shape (B * 3, F, T) representing the model outputs. First third are anchor samples, second third are positive samples, last third are negative samples.
        margin (float): Margin parameter for the triplet loss.
    Returns:
        torch.Tensor: Tensor of shape(B, T) representing the triplet loss for each time step.
    '''
    assert representations.dim() == 3, f"Representations must be a 3D tensor. Was shape {representations.shape}"
    assert representations.shape[0] % 3 == 0, f"First dimension of representations ({representations.shape[0]}) must be divisible by 3"
    B, F, T = representations.shape[0] // 3, representations.shape[1], representations.shape[2]
    assert representations.shape[0] == 3 * B, f"First dimension of representations ({representations.shape[0]}) must be three times the batch size ({3 * B})"
    ret = torch.zeros(B, T).to(representations.device)
    cos_sim = nn.CosineSimilarity(dim=1)
    for i in range(B):
        anchor = representations[i]
        positive = representations[i + B]
        negative = representations[i + (2 * B)]
        pos_sim = cos_sim(anchor.unsqueeze(0), positive.unsqueeze(0))
        neg_sim = cos_sim(anchor.unsqueeze(0), negative.unsqueeze(0))
        loss = torch.clamp(margin + neg_sim - pos_sim, min=0.0)
        ret[i] = loss
    return ret