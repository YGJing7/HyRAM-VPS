import sys
from os import path
import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hyram_vps.inference.data.vos_test_dataset import VOSTestDataset
from hyram_vps.model.hyram_vps import HyRAMVPS
from hyram_vps.inference.inference_core import InferenceCore
from hyram_vps.inference.utils.results_utils import ResultSaver
from hyram_vps.inference.utils.args_utils import get_dataset_cfg

from tqdm import tqdm

log = logging.getLogger()


@torch.inference_mode()
@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_sunseg_config.yaml')
def eval_vos(cfg: DictConfig):
    if cfg['output_dir'] is not None:
        run_dir = cfg['output_dir']
    else:
        run_dir = HydraConfig.get().run.dir
    log.info(f'All configuration: {cfg}')

    # Load the network weights
    hyram_vps = HyRAMVPS(cfg).cuda().eval()
    if cfg.weights is not None:
        model_weights = torch.load(cfg.weights)
        hyram_vps.load_weights(model_weights)
    else:
        log.warning('No model weights loaded. Are you sure about this?')

    dataset_name = cfg.dataset
    data_cfg = get_dataset_cfg(cfg)
    if dataset_name.startswith('sunseg') and data_cfg.use_all_masks:
        raise ValueError('SUN-SEG evaluation should use only the first-frame mask. '
                         'Set use_all_masks=False to avoid GT leakage.')
    if not dataset_name.startswith('sunseg-'):
        raise ValueError(f'Only SUN-SEG datasets are supported, got: {dataset_name}')

    # Set up the SUN-SEG dataset.
    image_dir = data_cfg.image_directory
    mask_dir = data_cfg.mask_directory
    meta_dataset = VOSTestDataset(image_dir,
                                  mask_dir,
                                  use_all_masks=False,
                                  size=data_cfg.size)
    use_amp = cfg.amp

    # multi-scale configurations
    save_scores = cfg.save_scores
    save_soft_scores = cfg.get('save_soft_scores', False)
    save_aux = cfg.save_aux
    if save_aux:
        # should have no effects on any other modules except on aux output
        hyram_vps = hyram_vps.train()

    # Set up loader
    meta_loader = meta_dataset.get_datasets()

    # determine where to save the masks
    save_all = data_cfg['save_all']
    mask_output_root = path.join(run_dir, 'Annotations')
    score_output_root = path.join(run_dir, 'Scores')
    soft_score_output_root = mask_output_root
    visualize_output_root = path.join(run_dir, 'Visualizations')

    total_process_time = 0
    total_frames = 0

    # Start eval
    pbar = tqdm(meta_loader, total=len(meta_dataset))
    for vid_reader in pbar:

        loader = DataLoader(vid_reader, batch_size=None, shuffle=False, num_workers=4)
        vid_name = vid_reader.vid_name
        pbar.set_description(vid_name)
        vid_length = len(loader)

        try:
            processor = InferenceCore(hyram_vps, cfg=cfg)
            saver = ResultSaver(mask_output_root,
                                vid_name,
                                dataset=dataset_name,
                                object_manager=processor.object_manager,
                                use_long_id=vid_reader.use_long_id,
                                palette=vid_reader.get_palette(),
                                save_mask=not save_soft_scores,
                                save_scores=save_scores,
                                score_output_root=score_output_root,
                                save_soft_scores=save_soft_scores,
                                soft_score_output_root=soft_score_output_root,
                                visualize_output_root=visualize_output_root,
                                visualize=cfg.visualize,
                                init_json=None)
            first_mask_loaded = False

            for ti, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    image = data['rgb'].cuda()
                    mask = data.get('mask')
                    if mask is not None:
                        mask = mask.cuda()
                    valid_labels = data.get('valid_labels')
                    if valid_labels is not None:
                        valid_labels = valid_labels.tolist()
                    info = data['info']
                    frame_name = info['frame']
                    shape = info['shape']
                    resize_needed = info['resize_needed']
                    path_to_image = info['path_to_image']

                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    # if, for some reason, the first frame is not aligned with the first mask
                    if not first_mask_loaded:
                        if mask is not None:
                            first_mask_loaded = True
                        else:
                            # no point to do anything without a mask
                            continue

                    # Run the model on this frame
                    prob = processor.step(image, mask, valid_labels, end=(ti == vid_length - 1))

                    end.record()
                    torch.cuda.synchronize()
                    total_process_time += (start.elapsed_time(end) / 1000)
                    total_frames += 1

                    if save_all or info['save']:
                        saver.process(prob,
                                      frame_name,
                                      resize_needed=resize_needed,
                                      shape=shape,
                                      last_frame=(ti == vid_length - 1),
                                      path_to_image=path_to_image)

            saver.end()
        except Exception as e:
            log.error(f'Runtime error at {vid_name}')
            log.error(e)
            saver.end()
            raise e

    log.info(f'Total processing time: {total_process_time}')
    log.info(f'Total processed frames: {total_frames}')
    log.info(f'FPS: {total_frames / total_process_time}')
    log.info(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

if __name__ == '__main__':
    eval_vos()
