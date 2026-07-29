"""SUN-SEG prediction writer."""

import os
from os import path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from hyram_vps.inference.object_manager import ObjectManager




def _attention_to_uint8(attn: np.ndarray) -> np.ndarray:
    attn = attn.astype(np.float32)
    lo, hi = np.percentile(attn, [1, 99])
    if hi <= lo:
        lo, hi = float(attn.min()), float(attn.max())
    if hi <= lo:
        return np.zeros_like(attn, dtype=np.uint8)
    attn = np.clip((attn - lo) / (hi - lo), 0, 1)
    return (attn * 255).astype(np.uint8)


def save_attention_overlays(
    output_root: str,
    video_name: str,
    frame_name: str,
    path_to_image: str,
    q_weights: torch.Tensor,
    object_manager: ObjectManager,
    *,
    object_id: int = 1,
    head: int = -1,
    alpha: float = 0.65,
) -> bool:
    """Save object-query cross-attention overlays for one frame.

    q_weights is expected to be B*num_objects*num_heads*num_queries*H*W.
    The default averages over heads and saves one blue overlay per query.
    """
    if path_to_image is None or q_weights is None or q_weights.ndim != 6:
        return False
    if object_id not in object_manager.obj_id_to_obj:
        if len(object_manager.tmp_id_to_obj) == 0:
            return False
        object_id = next(iter(object_manager.tmp_id_to_obj.values())).id

    tmp_id = object_manager.find_tmp_by_id(object_id)
    object_index = tmp_id - 1
    if object_index < 0 or object_index >= q_weights.shape[1]:
        return False

    weights = q_weights.detach().float().cpu()[0, object_index]
    if head >= 0:
        if head >= weights.shape[0]:
            return False
        weights = weights[head]
    else:
        weights = weights.mean(dim=0)

    image = Image.open(path_to_image).convert("RGB")
    image_np = np.array(image).astype(np.float32)
    frame_stem = path.splitext(frame_name)[0]
    frame_dir = path.join(output_root, video_name, frame_stem)
    os.makedirs(frame_dir, exist_ok=True)

    overlays = []
    for query_index, query_map in enumerate(weights):
        overlay_image = _make_blue_overlay(image_np, query_map.numpy(), alpha)
        overlay_image.save(path.join(frame_dir, f"query_{query_index:02d}.jpg"), quality=95)
        overlays.append(overlay_image)

    if overlays:
        cols = min(4, len(overlays))
        rows = int(np.ceil(len(overlays) / cols))
        sheet = Image.new("RGB", (image.width * cols, image.height * rows), (255, 255, 255))
        for index, overlay in enumerate(overlays):
            x = (index % cols) * image.width
            y = (index // cols) * image.height
            sheet.paste(overlay, (x, y))
        sheet.save(path.join(frame_dir, "queries_contact_sheet.jpg"), quality=95)

    return True

def _make_blue_overlay(image_np: np.ndarray, heat: np.ndarray, alpha: float) -> Image.Image:
    heat = _attention_to_uint8(heat)
    heat_image = Image.fromarray(heat).resize((image_np.shape[1], image_np.shape[0]),
                                              Image.Resampling.BILINEAR)
    heat_np = np.array(heat_image).astype(np.float32) / 255.0
    heat_np = heat_np[..., None]
    blue = np.zeros_like(image_np)
    blue[..., 0] = 80
    blue[..., 1] = 190
    blue[..., 2] = 255
    overlay = image_np * (1.0 - alpha * heat_np) + blue * (alpha * heat_np)
    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))


def save_region_summary_overlays(
    output_root: str,
    video_name: str,
    frame_name: str,
    path_to_image: str,
    region_aux: dict,
    object_manager: ObjectManager,
    *,
    object_id: int = 1,
    alpha: float = 0.65,
) -> bool:
    """Save boundary/interior/background summary masks and pooling-weight overlays."""
    if path_to_image is None or region_aux is None:
        return False
    if object_id not in object_manager.obj_id_to_obj:
        if len(object_manager.tmp_id_to_obj) == 0:
            return False
        object_id = next(iter(object_manager.tmp_id_to_obj.values())).id
    required = ['region_masks', 'summary_weights', 'region_names', 'region_slices']
    if any(key not in region_aux for key in required):
        return False

    region_masks = region_aux['region_masks']
    summary_weights = region_aux['summary_weights']
    if region_masks is None or summary_weights is None:
        return False
    if region_masks.ndim != 5 or summary_weights.ndim != 5:
        return False

    tmp_id = object_manager.find_tmp_by_id(object_id)
    object_index = tmp_id - 1
    if object_index < 0 or object_index >= region_masks.shape[1]:
        return False

    image = Image.open(path_to_image).convert("RGB")
    image_np = np.array(image).astype(np.float32)
    frame_stem = path.splitext(frame_name)[0]
    frame_dir = path.join(output_root, video_name, frame_stem)
    os.makedirs(frame_dir, exist_ok=True)

    region_maps = region_masks.detach().float().cpu()[0, object_index]
    weight_maps = summary_weights.detach().float().cpu()[0, object_index]
    names = list(region_aux['region_names'])
    slices = list(region_aux['region_slices'])

    contact_items = []
    for region_index, region_name in enumerate(names):
        if region_index >= region_maps.shape[-1]:
            continue
        region_map = region_maps[..., region_index].numpy()
        start, end = slices[region_index]
        if end > start:
            weight_map = weight_maps[..., start:end].mean(dim=-1).numpy()
        else:
            weight_map = np.zeros_like(region_map)

        mask_overlay = _make_blue_overlay(image_np, region_map, alpha)
        weight_overlay = _make_blue_overlay(image_np, weight_map, alpha)
        mask_overlay.save(path.join(frame_dir, f"region_{region_name}_mask.jpg"), quality=95)
        weight_overlay.save(path.join(frame_dir, f"region_{region_name}_weight.jpg"), quality=95)
        contact_items.extend([mask_overlay, weight_overlay])

    if contact_items:
        cols = len(names)
        rows = 2
        sheet = Image.new("RGB", (image.width * cols, image.height * rows), (255, 255, 255))
        for index, item in enumerate(contact_items):
            x = (index // rows) * image.width
            y = (index % rows) * image.height
            sheet.paste(item, (x, y))
        sheet.save(path.join(frame_dir, "region_contact_sheet.jpg"), quality=95)

    return bool(contact_items)

class ResultSaver:
    def __init__(
        self,
        output_root,
        video_name,
        *,
        dataset,
        object_manager: ObjectManager,
        use_long_id,
        palette=None,
        save_mask=True,
        save_scores=False,
        score_output_root=None,
        save_soft_scores=False,
        soft_score_output_root=None,
        visualize_output_root=None,
        visualize=False,
        init_json=None,
    ):
        del score_output_root, init_json
        if not dataset.startswith("sunseg-"):
            raise ValueError(f"Only SUN-SEG output is supported, got: {dataset}")
        if use_long_id:
            raise ValueError("SUN-SEG masks must be single-channel PNG files.")
        if save_scores:
            raise ValueError("Multi-scale score export is not part of the SUN-SEG pipeline.")

        self.output_root = output_root
        self.video_name = video_name
        self.object_manager = object_manager
        self.palette = palette
        self.save_mask = save_mask
        self.save_soft_scores = save_soft_scores
        self.soft_score_output_root = soft_score_output_root
        self.visualize_output_root = visualize_output_root
        self.visualize = visualize

    def process(
        self,
        prob: torch.Tensor,
        frame_name: str,
        resize_needed: bool = False,
        shape: Optional[Tuple[int, int]] = None,
        last_frame: bool = False,
        path_to_image: str = None,
    ):
        del last_frame
        if resize_needed:
            prob = F.interpolate(
                prob.unsqueeze(1), tuple(shape), mode="bilinear", align_corners=False
            )[:, 0]

        index_mask = torch.argmax(prob, dim=0)
        mask = torch.zeros_like(index_mask)
        for temporary_id, obj in self.object_manager.tmp_id_to_obj.items():
            mask[index_mask == temporary_id] = obj.id
        mask_array = mask.detach().cpu().numpy().astype(np.uint8)
        output_name = path.splitext(frame_name)[0] + ".png"

        if self.save_mask:
            output_dir = path.join(self.output_root, self.video_name)
            os.makedirs(output_dir, exist_ok=True)
            output_image = Image.fromarray(mask_array)
            if self.palette is not None:
                output_image.putpalette(self.palette)
            output_image.save(path.join(output_dir, output_name))

        if self.save_soft_scores:
            soft_dir = path.join(self.soft_score_output_root, self.video_name)
            os.makedirs(soft_dir, exist_ok=True)
            foreground = prob[1:].max(dim=0).values
            foreground = (foreground.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(foreground).save(path.join(soft_dir, output_name))

        if self.visualize:
            if path_to_image is None:
                raise ValueError("Visualization requires the source image path.")
            image = np.array(Image.open(path_to_image).convert("RGB"))
            overlay = image.copy()
            overlay[mask_array > 0] = (255, 0, 0)
            blended = (image * 0.6 + overlay * 0.4).astype(np.uint8)
            vis_dir = path.join(self.visualize_output_root, self.video_name)
            os.makedirs(vis_dir, exist_ok=True)
            Image.fromarray(blended).save(path.join(vis_dir, path.splitext(frame_name)[0] + ".jpg"))

    def end(self):
        return None
