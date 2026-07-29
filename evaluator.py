import glob
import os
from pathlib import Path
import cv2
import argparse
from tqdm import tqdm
try:
    import prettytable as pt
except ImportError:
    class _SimplePrettyTable:
        def __init__(self):
            self.field_names = []
            self._rows = []

        def add_row(self, row):
            self._rows.append(row)

        def __str__(self):
            rows = [self.field_names, *self._rows]
            widths = [max(len(str(row[i])) for row in rows) for i in range(len(self.field_names))]
            sep = '+-' + '-+-'.join('-' * width for width in widths) + '-+'

            def fmt(row):
                return '| ' + ' | '.join(str(value).ljust(width) for value, width in zip(row, widths)) + ' |'

            return '\n'.join([sep, fmt(self.field_names), sep, *(fmt(row) for row in self._rows), sep])

    class _PrettyTableModule:
        PrettyTable = _SimplePrettyTable

    pt = _PrettyTableModule()
import numpy as np

_EPS = np.spacing(1)
_TYPE = np.float64

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUNSEG_ROOT = Path("/mnt/b33c377d-a988-494e-860f-8149fffe7254/yangguojing/data/SUN-SEG")
DEFAULT_PRED_ROOT = PROJECT_ROOT / "output" / "sunseg_boundary_eval"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "eval-result"

DATASET_MAP = {
    "easy-seen": {
        "gt": Path("TestEasyDataset/Seen/GT"),
        "pred": Path("sunseg-easy-seen/Annotations"),
        "label": "TestEasyDataset/Seen",
    },
    "easy-unseen": {
        "gt": Path("TestEasyDataset/Unseen/GT"),
        "pred": Path("sunseg-easy-unseen/Annotations"),
        "label": "TestEasyDataset/Unseen",
    },
    "hard-seen": {
        "gt": Path("TestHardDataset/Seen/GT"),
        "pred": Path("sunseg-hard-seen/Annotations"),
        "label": "TestHardDataset/Seen",
    },
    "hard-unseen": {
        "gt": Path("TestHardDataset/Unseen/GT"),
        "pred": Path("sunseg-hard-unseen/Annotations"),
        "label": "TestHardDataset/Unseen",
    },
}
LEGACY_DATASET_TO_SPLIT = {cfg["label"]: split for split, cfg in DATASET_MAP.items()}


def _as_path(path_like):
    return Path(path_like).expanduser().resolve()


def _resolve_splits(data_lst):
    if not data_lst or "all" in data_lst:
        return list(DATASET_MAP.keys())
    splits = []
    for item in data_lst:
        split = LEGACY_DATASET_TO_SPLIT.get(item, item)
        if split not in DATASET_MAP:
            raise ValueError(f"Unknown split: {item}. Available: all, {list(DATASET_MAP)}")
        splits.append(split)
    return splits


def _prepare_data(pred, gt):
    gt = gt > 128
    pred = pred / 255.0
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _smeasure_per_image(pred_ary, gt_ary):
    pred, gt = _prepare_data(pred_ary, gt_ary)
    y = np.mean(gt)
    if y == 0:
        return 1.0 - np.mean(pred)
    elif y == 1:
        return np.mean(pred)

    alpha = 0.5
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    u = y

    # ---- s_object ----
    def _s_obj(in_pred, in_gt):
        mask = in_gt == 1
        c = np.count_nonzero(mask)
        if c == 0:
            return 0.0
        x = np.mean(in_pred[mask])
        sigma_x = np.std(in_pred[mask], ddof=1) if c > 1 else 0.0
        return 2.0 * x / (x * x + 1.0 + sigma_x + _EPS)

    object_score = u * _s_obj(fg, gt) + (1.0 - u) * _s_obj(bg, 1.0 - gt)

    # ---- centroid ----
    h, w = gt.shape
    area_obj = np.count_nonzero(gt)
    if area_obj == 0:
        cx, cy = int(np.round(w / 2.0)) + 1, int(np.round(h / 2.0)) + 1
    else:
        cy, cx = np.argwhere(gt).mean(axis=0).round()
        cx, cy = int(cx) + 1, int(cy) + 1

    # ---- ssim ----
    def _ssim(ip, ig):
        N = ip.size
        if N <= 1:
            return 1.0
        mx = np.mean(ip)
        my = np.mean(ig)
        sx = np.sum((ip - mx) ** 2) / (N - 1)
        sy = np.sum((ig - my) ** 2) / (N - 1)
        sxy = np.sum((ip - mx) * (ig - my)) / (N - 1)
        a = 4.0 * mx * my * sxy
        b = (mx * mx + my * my) * (sx + sy)
        if a != 0:
            return a / (b + _EPS)
        return 1.0 if (a == 0 and b == 0) else 0.0

    area = h * w
    w1 = cx * cy / area
    w2 = cy * (w - cx) / area
    w3 = (h - cy) * cx / area
    w4 = 1.0 - w1 - w2 - w3

    region_score = (
        w1 * _ssim(pred[0:cy, 0:cx], gt[0:cy, 0:cx]) +
        w2 * _ssim(pred[0:cy, cx:w], gt[0:cy, cx:w]) +
        w3 * _ssim(pred[cy:h, 0:cx], gt[cy:h, 0:cx]) +
        w4 * _ssim(pred[cy:h, cx:w], gt[cy:h, cx:w])
    )

    sm = alpha * object_score + (1.0 - alpha) * region_score
    return max(0.0, sm)


def _wfmeasure_per_image(pred_ary, gt_ary):
    from scipy.ndimage import convolve, distance_transform_edt as bwdist
    pred, gt = _prepare_data(pred_ary, gt_ary)
    if np.all(~gt):
        return 0.0

    Dst, Idxt = bwdist(gt == 0, return_indices=True)
    E = np.abs(pred - gt)
    Et = np.copy(E)
    Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

    # matlab_style_gauss2D(7,7, sigma=5)
    m, n = 3.0, 3.0
    yg, xg = np.ogrid[-m:m + 1, -n:n + 1]
    K = np.exp(-(xg * xg + yg * yg) / (2.0 * 5.0 * 5.0))
    K[K < np.finfo(K.dtype).eps * K.max()] = 0
    K /= K.sum()

    EA = convolve(Et, weights=K, mode="constant", cval=0)
    MIN_E_EA = np.where(gt & (EA < E), EA, E)
    B = np.where(gt == 0, 2.0 - np.exp(np.log(0.5) / 5.0 * Dst), np.ones_like(gt))
    Ew = MIN_E_EA * B

    TPw = float(np.sum(gt)) - float(np.sum(Ew[gt == 1]))
    FPw = float(np.sum(Ew[gt == 0]))
    R = 1.0 - float(np.mean(Ew[gt == 1]))
    P = TPw / (TPw + FPw + _EPS)
    beta = 1.0
    return (1.0 + beta) * R * P / (R + beta * P + _EPS)


def _mae_per_image(pred_ary, gt_ary):
    pred, gt = _prepare_data(pred_ary, gt_ary)
    return float(np.mean(np.abs(pred - gt)))


def _emeasure_per_image(pred_ary, gt_ary):
    pred, gt = _prepare_data(pred_ary, gt_ary)
    gt_fg_numel = np.count_nonzero(gt)
    gt_size = gt.size

    pred_bin = (pred * 255.0).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    fg_fg_hist, _ = np.histogram(pred_bin[gt], bins=bins)
    fg_bg_hist, _ = np.histogram(pred_bin[~gt], bins=bins)
    fg_fg = np.cumsum(np.flip(fg_fg_hist)).astype(np.float64)
    fg_bg = np.cumsum(np.flip(fg_bg_hist)).astype(np.float64)

    fg_all = fg_fg + fg_bg
    bg_all = float(gt_size) - fg_all

    # Handle edge cases (no foreground or no background)
    if gt_fg_numel == 0:
        enhanced_sum = bg_all
    elif gt_fg_numel == gt_size:
        enhanced_sum = fg_all
    else:
        mean_pred_val = fg_all / float(gt_size)
        mean_gt_val = float(gt_fg_numel) / float(gt_size)

        d_pr_fg = 1.0 - mean_pred_val
        d_pr_bg = 0.0 - mean_pred_val
        d_gt_fg = np.full(256, 1.0 - mean_gt_val)
        d_gt_bg = np.full(256, 0.0 - mean_gt_val)

        bg_fg = float(gt_fg_numel) - fg_fg
        bg_bg = bg_all - bg_fg

        parts = np.stack([fg_fg, fg_bg, bg_fg, bg_bg])
        combs = np.stack([
            np.stack([d_pr_fg, d_gt_fg]),
            np.stack([d_pr_fg, d_gt_bg]),
            np.stack([d_pr_bg, d_gt_fg]),
            np.stack([d_pr_bg, d_gt_bg]),
        ])

        align = 2.0 * combs[:, 0] * combs[:, 1] / (
            combs[:, 0] ** 2 + combs[:, 1] ** 2 + _EPS)
        enhanced = (align + 1.0) ** 2 / 4.0
        enhanced_sum = np.sum(enhanced * parts, axis=0)

    em_curve = enhanced_sum / (float(gt_size) - 1.0 + _EPS)

    # Adaptive E-measure: compute directly (matching original cal_em_with_threshold)
    thresh = min(2.0 * float(np.mean(pred)), 1.0)
    bin_pred = pred >= thresh
    fg_fg_t = float(np.count_nonzero(bin_pred & gt))
    fg_bg_t = float(np.count_nonzero(bin_pred & ~gt))
    fg_all_t = fg_fg_t + fg_bg_t
    bg_all_t = float(gt_size) - fg_all_t

    if gt_fg_numel == 0:
        adp_em = bg_all_t / (float(gt_size) - 1.0 + _EPS)
    elif gt_fg_numel == gt_size:
        adp_em = fg_all_t / (float(gt_size) - 1.0 + _EPS)
    else:
        mean_pred_t = fg_all_t / float(gt_size)
        mean_gt_t = float(gt_fg_numel) / float(gt_size)
        d_gt_fg_t = 1.0 - mean_gt_t
        d_gt_bg_t = 0.0 - mean_gt_t
        d_pr_fg_t = 1.0 - mean_pred_t
        d_pr_bg_t = 0.0 - mean_pred_t
        bg_fg_t = float(gt_fg_numel) - fg_fg_t
        bg_bg_t = bg_all_t - bg_fg_t

        combs_t = [
            (d_pr_fg_t, d_gt_fg_t), (d_pr_fg_t, d_gt_bg_t),
            (d_pr_bg_t, d_gt_fg_t), (d_pr_bg_t, d_gt_bg_t),
        ]
        parts_t = [fg_fg_t, fg_bg_t, bg_fg_t, bg_bg_t]
        total = 0.0
        for (c0, c1), p in zip(combs_t, parts_t):
            av = 2.0 * c0 * c1 / (c0 ** 2 + c1 ** 2 + _EPS)
            total += ((av + 1.0) ** 2 / 4.0) * p
        adp_em = total / (float(gt_size) - 1.0 + _EPS)

    return {'adpEm': float(adp_em), 'em_curve': em_curve}


def _fmeasure_per_image(pred_ary, gt_ary):
    pred, gt = _prepare_data(pred_ary, gt_ary)
    beta = 0.3

    # adaptive Fm
    thresh = min(2.0 * float(np.mean(pred)), 1.0)
    bin_pred = pred >= thresh
    inter = float(np.count_nonzero(bin_pred & gt))
    if inter == 0:
        adp_fm = 0.0
    else:
        pre = inter / max(float(np.count_nonzero(bin_pred)), 1.0)
        rec = inter / max(float(np.count_nonzero(gt)), 1.0)
        adp_fm = (1.0 + beta) * pre * rec / (beta * pre + rec)

    # changeable Fm (vectorised across 256 thresholds)
    pred_bin = (pred * 255.0).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    fg_hist, _ = np.histogram(pred_bin[gt], bins=bins)
    bg_hist, _ = np.histogram(pred_bin[~gt], bins=bins)
    TPs = np.cumsum(np.flip(fg_hist)).astype(np.float64)
    Ps = TPs + np.cumsum(np.flip(bg_hist)).astype(np.float64)
    Ps[Ps == 0] = 1.0
    T = max(float(np.count_nonzero(gt)), 1.0)
    prec = TPs / Ps
    rec = TPs / T
    num = (1.0 + beta) * prec * rec
    den = np.where(num == 0, 1.0, beta * prec + rec)
    fm_curve = num / den

    return {'adpFm': adp_fm, 'fm_curve': fm_curve}


def _medical_per_image(pred_ary, gt_ary):
    pred, gt = _prepare_data(pred_ary, gt_ary)
    Thresholds = np.linspace(1, 0, 256, dtype=np.float64)  # descending

    num_obj = float(np.count_nonzero(gt))
    total = float(gt.size)
    bg_pixels = total - num_obj

    if num_obj == 0:
        zero256 = np.zeros(256, dtype=np.float64)
        return {'threshold_Sensitivity': zero256.copy(),
                'threshold_Specificity': zero256.copy(),
                'threshold_Dice': zero256.copy(),
                'threshold_IoU': zero256.copy()}

    # Sort for exact count-above-threshold via binary search
    sorted_all = np.sort(pred.ravel())
    sorted_fg = np.sort(pred[gt])

    NumRec = (total - np.searchsorted(sorted_all, Thresholds, side='left')).astype(np.float64)
    NumAnd = (num_obj - np.searchsorted(sorted_fg, Thresholds, side='left')).astype(np.float64)

    FN = num_obj - NumAnd

    Recall = np.where(NumAnd == 0, 0.0, NumAnd / num_obj)
    TN = total - NumRec - FN
    Specificity = np.where(bg_pixels == 0, 0.0, TN / bg_pixels)
    denom = num_obj + NumRec - NumAnd
    IoU = np.where(NumAnd == 0, 0.0, NumAnd / np.maximum(denom, 1.0))
    Dice = np.where(NumAnd == 0, 0.0, 2.0 * NumAnd / (num_obj + NumRec))

    return {
        'threshold_Sensitivity': Recall,
        'threshold_Specificity': Specificity,
        'threshold_Dice': Dice,
        'threshold_IoU': IoU,
    }


_METRIC_FUNC = {
    'Smeasure':         ('scalar', _smeasure_per_image),
    'WeightedFmeasure': ('scalar', _wfmeasure_per_image),
    'MAE':              ('scalar', _mae_per_image),
    'Emeasure':         ('curve',  _emeasure_per_image),
    'Fmeasure':         ('curve',  _fmeasure_per_image),
    'Medical':          ('curve',  _medical_per_image),
}


def _process_batch(args):
    """
    Worker: process a batch of images, return aggregated metrics
    (one dict per metric_module * metric_name).
    """
    gt_paths, pred_paths, metric_module_names = args
    n = len(gt_paths)

    # Instantiate accumulators
    accum = {}
    for mname in metric_module_names:
        kind, _ = _METRIC_FUNC[mname]
        if kind == 'scalar':
            accum[mname] = 0.0
        else:
            accum[mname] = np.zeros(256, dtype=np.float64)

    for gt_pth, pred_pth in zip(gt_paths, pred_paths):
        pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)
        gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)
        if pred_ary is None or gt_ary is None:
            raise FileNotFoundError(f"Missing: gt={gt_pth}  pred={pred_pth}")
        if gt_ary.shape != pred_ary.shape:
            pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        for mname in metric_module_names:
            _, func = _METRIC_FUNC[mname]
            result = func(pred_ary, gt_ary)
            if isinstance(result, dict):
                # full metric names so we can aggregate later.
                for sub_k, sub_v in result.items():
                    full_k = f'{mname}__{sub_k}'
                    if full_k not in accum:
                        accum[full_k] = np.zeros(256, dtype=np.float64)
                    accum[full_k] += sub_v
            else:
                accum[mname] += float(result)

    # Divide by batch size for scalars
    for k, v in accum.items():
        if np.ndim(v) == 0:
            accum[k] = v / n
        else:
            accum[k] = v / n
    accum['_batch_size'] = n
    return accum


def evaluator(gt_pth_lst, pred_pth_lst, metrics, n_workers=1):
    """
    Parameters
    ----------
    n_workers : int
        Number of parallel workers.  Use 0 or a negative number for
        ``min(CPU count, num_images)``.
    """
    module_map_name = {
        "Smeasure": "Smeasure", "wFmeasure": "WeightedFmeasure", "MAE": "MAE",
        "adpEm": "Emeasure", "meanEm": "Emeasure", "maxEm": "Emeasure",
        "adpFm": "Fmeasure", "meanFm": "Fmeasure", "maxFm": "Fmeasure",
        "meanSen": "Medical", "maxSen": "Medical",
        "meanSpe": "Medical", "maxSpe": "Medical",
        "meanDice": "Medical", "maxDice": "Medical",
        "meanIoU": "Medical", "maxIoU": "Medical",
    }

    assert len(gt_pth_lst) == len(pred_pth_lst), \
        f"gt / pred length mismatch: {len(gt_pth_lst)} vs {len(pred_pth_lst)}"
    N = len(gt_pth_lst)
    if N == 0:
        return {}

    metric_module_names = sorted(set(module_map_name[m] for m in metrics))

    if n_workers <= 0:
        import multiprocessing as mp
        n_workers = min(mp.cpu_count(), N)
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        batch_args = (gt_pth_lst, pred_pth_lst, metric_module_names)
        batch_res = _process_batch(batch_args)
    else:
        import multiprocessing as mp
        batch_size = max(1, (N + n_workers - 1) // n_workers)
        batches = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batches.append((
                gt_pth_lst[start:end],
                pred_pth_lst[start:end],
                metric_module_names,
            ))

        ctx = mp.get_context('spawn' if os.name == 'nt' else 'fork')
        with ctx.Pool(processes=min(n_workers, len(batches))) as pool:
            batch_results = pool.map(_process_batch, batches)

        sizes = [r.pop('_batch_size') for r in batch_results]
        total = sum(sizes)

        batch_res = {}
        for key in batch_results[0].keys():
            if key == '_batch_size':
                continue
            weighted_sum = sum(r[key] * s for r, s in zip(batch_results, sizes))
            batch_res[key] = weighted_sum / total

    res = {}
    for metric in metrics:
        mod = module_map_name[metric]
        if mod in ('Smeasure', 'WeightedFmeasure', 'MAE'):
            res[metric] = batch_res[mod]                          # scalar
        elif mod == 'Emeasure':
            res[metric] = batch_res['Emeasure__em_curve']        # 256鈥慳rray
        elif mod == 'Fmeasure':
            res[metric] = batch_res['Fmeasure__fm_curve']        # 256鈥慳rray
        elif mod == 'Medical':
            key_map = {
                'Sen': 'Sensitivity', 'Spe': 'Specificity',
                'Dice': 'Dice', 'IoU': 'IoU',
            }
            for suffix, skey in key_map.items():
                if suffix in metric:
                    res[metric] = batch_res[f'Medical__threshold_{skey}']  # 256鈥慳rray
                    break

    return res


def get_competitors(root):
    for model_name in os.listdir(root):
        print('\'{}\''.format(model_name), end=', ')


def _natural_key(name):
    """Fast natural鈥憇ort key that avoids the fragile try/except chain."""
    import re
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def eval_engine_vps(opt, txt_save_path):
    gt_root = _as_path(opt.gt_root)
    pred_root = _as_path(opt.pred_root)
    splits = _resolve_splits(opt.data_lst)

    for split in splits:
        cfg = DATASET_MAP[split]
        data_label = cfg["label"]
        print('#' * 20, 'Current Dataset:', split, '#' * 20)
        filename = os.path.join(txt_save_path, '{}_eval.txt'.format(split))

        gt_src = gt_root / cfg["gt"]
        pred_src = pred_root / cfg["pred"]
        if not gt_src.is_dir():
            raise FileNotFoundError(f"GT directory not found: {gt_src}")
        if not pred_src.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {pred_src}")

        with open(filename, 'w+', encoding='utf-8') as file_to_write:
            tb = pt.PrettyTable()
            names = ["Dataset", "Method"]
            names.extend(opt.metric_list)
            tb.field_names = names

            method_name = opt.method_name
            print('#' * 10, 'Current Method:', method_name, '#' * 10)

            case_list = sorted(
                [x for x in os.listdir(gt_src) if 'DS_Store' not in x],
                key=_natural_key,
            )

            case_results = []
            for case in tqdm(case_list, desc=f'  {method_name}'):
                case_gt_name_list = sorted(
                    glob.glob(str(gt_src / case / '*.png')),
                    key=_natural_key,
                )
                case_pred_name_list = [
                    str(pred_src / case / Path(p).name) for p in case_gt_name_list
                ]

                missing = [p for p in case_pred_name_list if not os.path.isfile(p)]
                if missing:
                    raise FileNotFoundError(
                        f"Missing {len(missing)} prediction(s) for {split}/{case}; "
                        f"first missing: {missing[0]}"
                    )

                result = evaluator(
                    gt_pth_lst=case_gt_name_list,
                    pred_pth_lst=case_pred_name_list,
                    metrics=opt.metric_list,
                    n_workers=getattr(opt, 'n_workers', 1),
                )
                case_results.append(result)

            case_score_list = []
            for metric_name in opt.metric_list:
                values = [cr[metric_name] for cr in case_results]
                if 'max' in metric_name:
                    case_score_list.append(float(np.mean(values, axis=0).max().round(3)))
                elif 'mean' in metric_name:
                    case_score_list.append(float(np.mean(values, axis=0).mean().round(3)))
                else:
                    case_score_list.append(float(np.mean(values).round(3)))

            final_score_list = ['{:.3f}'.format(c) for c in case_score_list]
            tb.add_row([data_label, method_name] + final_score_list)

            print(tb)
            file_to_write.write(str(tb))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str,
        default=str(DEFAULT_SUNSEG_ROOT),
        help='SUN-SEG dataset root.')
    parser.add_argument(
        '--metric_list', nargs='+',
        default=['Smeasure', 'meanEm', 'wFmeasure', 'meanFm',
                 'meanSen', 'maxDice'],
        choices=['Smeasure', 'wFmeasure', 'MAE', 'adpEm', 'meanEm', 'maxEm',
                 'adpFm', 'meanFm', 'maxFm',
                 'meanSen', 'maxSen', 'meanSpe', 'maxSpe',
                 'meanDice', 'maxDice', 'meanIoU', 'maxIoU'])
    parser.add_argument(
        '--data_lst', '--splits', dest='data_lst', nargs='+',
        default=['all'],
        choices=['all', *DATASET_MAP.keys(),
                 'TestEasyDataset/Seen', 'TestHardDataset/Seen',
                 'TestEasyDataset/Unseen', 'TestHardDataset/Unseen'],
        help='SUN-SEG splits to evaluate.')
    parser.add_argument(
        '--pred_root', type=str,
        default=str(DEFAULT_PRED_ROOT),
        help='Prediction root containing sunseg-*/Annotations directories.')
    parser.add_argument('--method_name', type=str, default='Ours')
    parser.add_argument('--txt_name', type=str, default='vps_eval_result')
    parser.add_argument('--check_integrity', action='store_true')
    parser.add_argument(
        '--n_workers', type=int, default=0,
        help='Number of parallel workers (0 = auto, 1 = sequential).')
    opt = parser.parse_args()

    print('#' * 10, opt.data_lst)

    pred_name = Path(opt.pred_root).name
    txt_save_path = DEFAULT_RESULT_ROOT / opt.txt_name / pred_name
    os.makedirs(txt_save_path, exist_ok=True)

    if opt.check_integrity:
        gt_root = _as_path(opt.gt_root)
        pred_root = _as_path(opt.pred_root)
        for split in _resolve_splits(opt.data_lst):
            cfg = DATASET_MAP[split]
            gt_pth = gt_root / cfg['gt']
            pred_pth = pred_root / cfg['pred']
            gt_cases = sorted([p.name for p in gt_pth.iterdir() if p.is_dir()])
            pred_cases = sorted([p.name for p in pred_pth.iterdir() if p.is_dir()])
            if gt_cases != pred_cases:
                print(f'  Mismatch: {split}')
    else:
        print('>>> Skip check the integrity of each candidates ...')

    eval_engine_vps(opt, txt_save_path)

