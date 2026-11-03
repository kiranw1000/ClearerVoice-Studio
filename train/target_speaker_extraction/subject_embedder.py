# Based on FiLM. Learn subject embeddings and use them to modulate the EEG features in the neuroheed model.

from enum import Enum

import wandb

from tqdm import tqdm
from torch import nn
from solver import Solver
import torch
from .models.neuroheed.neuroheed import neuroheed


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
    
class Phase1Solver(Solver):
    def __init__(self, args, model, optimizer, train_data, validation_data, test_data):
        super().__init__(args, model, optimizer, train_data, validation_data, test_data)
        self.subject_embedding_type = args.subject_embedding_type
        self.model.freeze_encoder()
        self.model.freeze_separator()
        self.model.freeze_decoder()
        
    def _run_one_epoch(self, data_loader, state="train"):
        total_loss = 0
        self.accu_count = 0
        self.optimizer.zero_grad()
        for i, (eeg, a_mix, a_tgt, a_unatt, subject_id) in enumerate(tqdm(data_loader)):
            est_mask = self.model(a_mix, eeg, subject_id = subject_id)
            a_tgt_dif = torch.clamp(a_tgt / a_mix, min=1e-6, max=1.0)
            a_unatt_dif = torch.clamp(a_unatt / a_mix, min=1e-6, max=1.0)
            a_tgt_dif_mask = self.model.encoder(a_tgt_dif)
            a_unatt_dif_mask = self.model.encoder(a_unatt_dif)
            loss = self.loss(est_mask, a_tgt_dif_mask, a_unatt_dif_mask)
            wandb.log({"train_loss": loss}, step=self.global_step) if state=='train' and self.args.wandb and (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed else None
            if state == "train":
                if self.args.accu_grad:
                    self.accu_count += 1
                    loss_scaled = loss/(self.args.effec_batch_size / self.args.batch_size)
                    loss_scaled.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        if self.args.lr_warmup: self._adjust_lr_warmup()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.args.lr_warmup: self._adjust_lr_warmup()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        return total_loss / (i+1)
                    
class Phase2Solver(Solver):
    def __init__(self, args, model, optimizer, train_data, validation_data, test_data):
        super().__init__(args, model, optimizer, train_data, validation_data, test_data)
        self.subject_embedding_type = args.subject_embedding_type
        self.model.freeze_encoder()
        self.model.freeze_separator()
        self.model.freeze_decoder()
        
    def _run_one_epoch(self, data_loader, state="train"):
        total_loss = 0
        self.accu_count = 0
        self.optimizer.zero_grad()
        for i, (eeg, a_mix, a_tgt, subject_id) in enumerate(tqdm(data_loader)):
            est_mask = self.model(a_mix, eeg, subject_id = subject_id)
            a_tgt_dif = torch.clamp(a_tgt / a_mix, min=1e-6, max=1.0)
            a_tgt_dif_mask = self.model.encoder(a_tgt_dif)
            loss = self.loss(est_mask, a_tgt_dif_mask)
            wandb.log({"train_loss": loss}, step=self.global_step) if state=='train' and self.args.wandb and (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed else None
            if state == "train":
                if self.args.accu_grad:
                    self.accu_count += 1
                    loss_scaled = loss/(self.args.effec_batch_size / self.args.batch_size)
                    loss_scaled.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        if self.args.lr_warmup: self._adjust_lr_warmup()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.args.lr_warmup: self._adjust_lr_warmup()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        return total_loss / (i+1)
    
class Phase3Solver(Solver):
    def __init__(self, args, model, optimizer, train_data, validation_data, test_data):
        super().__init__(args, model, optimizer, train_data, validation_data, test_data)
        self.subject_embedding_type = args.subject_embedding_type
        self.model.freeze_subject_embeddings()
        
    def _run_one_epoch(self, data_loader, state="train"):
        total_loss = 0
        self.accu_count = 0
        self.optimizer.zero_grad()
        for i, (eeg, a_mix, a_tgt, subject_id) in enumerate(tqdm(data_loader)):
            a_tgt_est = self.model(a_mix, eeg, subject_id = subject_id)
            loss = self.loss(a_tgt_est, a_tgt)
            wandb.log({"train_loss": loss}, step=self.global_step) if state=='train' and self.args.wandb and (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed else None
            if state == "train":
                if self.args.accu_grad:
                    self.accu_count += 1
                    loss_scaled = loss/(self.args.effec_batch_size / self.args.batch_size)
                    loss_scaled.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        if self.args.lr_warmup: self._adjust_lr_warmup()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.args.lr_warmup: self._adjust_lr_warmup()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        return total_loss / (i+1)