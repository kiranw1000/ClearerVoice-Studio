# Based on FiLM. Learn subject embeddings and use them to modulate the EEG features in the neuroheed model.

from enum import Enum
from torch import nn
import torch


class SubjectEmbeddingType(Enum):
    """
    Enum for subject embedding types.
    """
    VECTOR = 1
    MATRIX = 2
    FILM = 3

class SubjectEncodings(nn.Module):
    """
    Learnable subject embeddings
    """
    
    
    
    def __init__(self, num_subjects: int, d_model: int, embedding_type: SubjectEmbeddingType = SubjectEmbeddingType.VECTOR):
        """Create subject embeddings.

        Args:
            num_subjects (int): number of subjects to choose from
            d_model (int): dimensionality of the subject eeg embeddings
            embedding_type (SubjectEmbeddingType): type of subject embeddings (VECTOR or MATRIX)
        """
        super().__init__()
        self.embedding_type = embedding_type
        self.subject_embeddings = nn.Embedding(num_subjects, d_model)
        
    def set_subject_embeddings(self, subject_embeddings: nn.Embedding):
        """Create subject embeddings.

        Args:
            subject_embeddings (nn.Embedding): pre-trained subject embeddings
        """
        self.subject_embeddings = subject_embeddings

    def forward(self, subject_id:int, input:torch.Tensor):
        """
        Args:
            subject_ids: Tensor of shape [batch_size] containing subject IDs
            input: Tensor of shape [batch_size, d_model] containing input features
        Returns:
            Tensor of shape [batch_size, d_model] containing subject embeddings
        """
        if self.embedding_type == SubjectEmbeddingType.VECTOR:
            return self.subject_embeddings(subject_id) + input
        elif self.embedding_type == SubjectEmbeddingType.MATRIX:
            return torch.matmul(self.subject_embeddings(subject_id).unsqueeze(1), input)

class FiLMNetwork(SubjectEncodings):
    def __init__(self, num_subjects:int, d_model:int):
        """
        Args:
            num_subjects (int): number of subjects
            d_model (int): dimension of the model
        """
        super().__init__(num_subjects, d_model)
        self.gamma = nn.Embedding(num_subjects, d_model)
        self.beta = nn.Embedding(num_subjects, d_model)

    def forward(self, subject_id:int, input:torch.Tensor):
        """
        Args:
            subject_id: Tensor of shape [batch_size] containing subject IDs
            input: Tensor of shape [batch_size, d_model] containing input features
        Returns:
            Tensor of shape [batch_size, d_model] containing FiLMed output
        """
        gamma = self.gamma(subject_id)
        beta = self.beta(subject_id)
        return gamma * input + beta