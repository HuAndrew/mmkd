# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmrazor.registry import MODELS


def mask_l2_loss(
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        saptial_attention_mask: Optional[torch.Tensor] = None,
        channel_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

    diff = (tensor_a - tensor_b)**2
    if saptial_attention_mask is not None:
        diff = diff * saptial_attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask
    diff = torch.sum(diff)**0.5
    return diff


@MODELS.register_module()
class FBRDLoss(nn.Module):

    def __init__(self,
                 mask_l2_weight: float = 7e-5,
                 channel_weight: float = 4e-3,
                 spatial_weight: float = 4e-3,
                 nonloacl_weight: float = 7e-5,
                 loss_weight: float = 1.0) -> None:
        """Inits FBKDLoss."""
        super().__init__()

        self.mask_l2_weight = mask_l2_weight
        self.channel_weight = channel_weight
        self.spatial_weight = spatial_weight
        self.nonloacl_weight = nonloacl_weight
        self.loss_weight = loss_weight

    def forward(self, s_input: Tuple[torch.Tensor, ...],
                t_input: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        losses = 0.0

        (s_spatial_mask, s_channel_mask, s_channel_pool_adapt,
         s_spatial_pool_adapt, s_relation_adapt, s_feat_adapt) = s_input

        (t_spatial_mask, t_channel_mask, t_spatial_pool, t_relation,
         t_feat) = t_input

        # Spatial-wise mask.
        spatial_sum_mask = (t_spatial_mask + s_spatial_mask) / 2
        spatial_sum_mask = spatial_sum_mask.detach()

        # Channel-wise mask, but not used in the FBKD source code.
        channel_sum_mask = (t_channel_mask + s_channel_mask) / 2
        channel_sum_mask = channel_sum_mask.detach()

        # feat_loss with mask
        losses += mask_l2_loss(
            t_feat,
            s_feat_adapt,
            saptial_attention_mask=spatial_sum_mask,
            channel_attention_mask=None) * self.mask_l2_weight

        # channel_loss
        losses += torch.dist(torch.mean(t_feat, [2, 3]),
                             s_channel_pool_adapt) * self.channel_weight

        # spatial_loss
        losses += torch.dist(t_spatial_pool,
                             s_spatial_pool_adapt) * self.spatial_weight

        # nonlocal_loss
        losses += torch.dist(
            t_relation, s_relation_adapt, p=2) * self.nonloacl_weight

        return self.loss_weight * losses
