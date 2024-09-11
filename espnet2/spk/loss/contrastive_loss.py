#! /usr/bin/python
# -*- encoding: utf-8 -*-

import numpy as np
from espnet2.spk.loss.abs_loss import AbsLoss

import torch
import torch.nn.functional as F


class Contrastive(AbsLoss):
    """SimCLR contrastive loss

    Paper: "CONTRASTIVE SELF-SUPERVISED LEARNING FOR TEXT-INDEPENDENT SPEAKER VERIFICATION"

    args:

    """

    def __init__(
        self,
        nout,
        temp: float = 0.5,
        weight1: float = 0.3,
        weight2: float = 0.3,
        weight3: float = 0.3,
        **kwargs,
    ):
        super().__init__(nout)
        self.temp = temp
        self.weight1 = weight1  # weight for spk
        self.weight2 = weight2  # weight for pitch1
        self.weight3 = weight3  # weight for pitch2

    def forward(self, x, intensities, label=None):
        temp = self.temp

        # x: (2N, nout)
        embeddings = x
        batch_size = embeddings.size(0)

        # Normalize the embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        # Create mask to remove self-similarities from the similarity matrix
        mask = torch.eye(
            similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device
        )
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Create labels for the positive pairs
        labels = torch.cat(
            [
                torch.arange(batch_size // 2) + (batch_size // 2) - 1,
                torch.arange(batch_size // 2),
            ],
            dim=0,
        )
        labels = labels.to(similarity_matrix.device)

        # Compute logits
        if intensities is None:
            logits = similarity_matrix / temp
        else:
            assert (
                intensities.shape[0] == batch_size // 2
            ), f"contrastive_loss.py: intensities batch_size({intensities.shape[0]}) should be half of input batch_size({batch_size})!"
            intensities = np.concatenate((intensities, intensities))

            intensities = torch.Tensor(np.abs(intensities)).to(similarity_matrix.device)
            margin = torch.zeros((batch_size, batch_size - 1)).to(similarity_matrix.device)
            row_indices = torch.arange(batch_size).to(similarity_matrix.device)
            margin[row_indices, labels] = intensities
            logits = (similarity_matrix + margin) / temp

        # Compute the loss
        loss = F.cross_entropy(logits, labels)

        return loss
