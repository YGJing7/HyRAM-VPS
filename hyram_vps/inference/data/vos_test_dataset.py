import os
from os import path
from typing import Iterable

from hyram_vps.inference.data.video_reader import VideoReader


class VOSTestDataset:
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 *,
                 use_all_masks: bool,
                 size: int = -1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.use_all_masks = use_all_masks
        self.size = size
        self.vid_list = sorted(os.listdir(self.mask_dir))

    def get_datasets(self) -> Iterable[VideoReader]:
        for video in self.vid_list:
            yield VideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                size=self.size,
                use_all_masks=self.use_all_masks,
            )

    def __len__(self):
        return len(self.vid_list)
