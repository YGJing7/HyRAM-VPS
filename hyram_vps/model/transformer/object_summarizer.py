from typing import List, Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from hyram_vps.model.transformer.positional_encoding import PositionalEncoding


# @torch.jit.script
def _weighted_pooling(masks: torch.Tensor, value: torch.Tensor,
                      logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    # value: B*num_objects*H*W*value_dim
    # logits: B*num_objects*H*W*num_summaries
    # masks: B*num_objects*H*W*num_summaries: 1 if allowed
    weights = logits.sigmoid() * masks
    # B*num_objects*num_summaries*value_dim
    sums = torch.einsum('bkhwq,bkhwc->bkqc', weights, value)
    # B*num_objects*H*W*num_summaries -> B*num_objects*num_summaries*1
    area = weights.flatten(start_dim=2, end_dim=3).sum(2).unsqueeze(-1)

    # B*num_objects*num_summaries*value_dim
    return sums, area


class ObjectSummarizer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_summarizer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.region_mode = this_cfg.get('region_mode', 'foreground_background')
        self.boundary_kernel_size = this_cfg.get('boundary_kernel_size', 3)
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature

        if self.add_pe:
            self.pos_enc = PositionalEncoding(self.embed_dim,
                                              scale=self.pixel_pe_scale,
                                              temperature=self.pixel_pe_temperature)

        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_summaries),
        )

    def _repeat_region_masks(self, regions: List[torch.Tensor]) -> torch.Tensor:
        num_regions = len(regions)
        base_count = self.num_summaries // num_regions
        remainder = self.num_summaries % num_regions

        repeated_regions = []
        for i, region in enumerate(regions):
            repeat_count = base_count + (1 if i < remainder else 0)
            if repeat_count > 0:
                repeated_regions.append(region.expand(-1, -1, -1, -1, repeat_count))

        return torch.cat(repeated_regions, dim=-1)

    def _get_region_names(self) -> List[str]:
        if self.region_mode == 'boundary_interior_background':
            return ['boundary', 'interior', 'background']
        if self.region_mode == 'none':
            return ['all']
        return ['foreground', 'background']

    def _get_region_slices(self, num_regions: int) -> List[tuple]:
        base_count = self.num_summaries // num_regions
        remainder = self.num_summaries % num_regions
        slices = []
        start = 0
        for i in range(num_regions):
            count = base_count + (1 if i < remainder else 0)
            slices.append((start, start + count))
            start += count
        return slices

    def _make_boundary_regions(self, masks: torch.Tensor) -> List[torch.Tensor]:
        # Regions generated online at the memory feature resolution.
        # We use max-pooling morphology to avoid requiring extra boundary labels.
        batch_size, num_objects, h, w, _ = masks.shape
        masks_4d = masks.squeeze(-1).flatten(0, 1).unsqueeze(1)

        kernel_size = self.boundary_kernel_size
        padding = kernel_size // 2
        dilated = F.max_pool2d(masks_4d, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-masks_4d, kernel_size, stride=1, padding=padding)

        boundary_in = (masks_4d - eroded).clamp(0, 1)
        boundary_out = (dilated - masks_4d).clamp(0, 1)
        boundary = (boundary_in + boundary_out).clamp(0, 1)
        interior = (masks_4d - boundary_in).clamp(0, 1)
        background = (1 - dilated).clamp(0, 1)

        return [
            boundary.view(batch_size, num_objects, h, w, 1),
            interior.view(batch_size, num_objects, h, w, 1),
            background.view(batch_size, num_objects, h, w, 1),
        ]

    def forward(self,
                masks: torch.Tensor,
                value: torch.Tensor,
                need_weights: bool = False) -> (torch.Tensor, Optional[torch.Tensor]):
        # masks: B*num_objects*(H0)*(W0)
        # value: B*num_objects*value_dim*H*W
        # -> B*num_objects*H*W*value_dim
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode='area')
        masks = masks.unsqueeze(-1)
        if self.region_mode == 'boundary_interior_background':
            regions = self._make_boundary_regions(masks)
        elif self.region_mode == 'none':
            regions = [torch.ones_like(masks)]
        else:
            inv_masks = 1 - masks
            regions = [masks, inv_masks]
        repeated_masks = self._repeat_region_masks(regions)

        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe

        with torch.cuda.amp.autocast(enabled=False):
            value = value.float()
            feature = self.feature_pred(value)
            logits = self.weights_pred(value)
            sums, area = _weighted_pooling(repeated_masks, feature, logits)

        summaries = torch.cat([sums, area], dim=-1)

        if need_weights:
            aux = {
                'summary_logits': logits,
                'summary_weights': logits.sigmoid() * repeated_masks,
                'region_masks': torch.cat(regions, dim=-1),
                'region_names': self._get_region_names(),
                'region_slices': self._get_region_slices(len(regions)),
            }
            return summaries, aux
        else:
            return summaries, None
