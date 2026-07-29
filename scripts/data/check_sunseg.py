from argparse import ArgumentParser
from pathlib import Path


SPLITS = (
    ("TrainDataset",),
    ("TestEasyDataset", "Seen"),
    ("TestEasyDataset", "Unseen"),
    ("TestHardDataset", "Seen"),
    ("TestHardDataset", "Unseen"),
)


def paired_roots(root: Path, split):
    base = root.joinpath(*split)
    return base / "Frame", base / "GT"


def frame_stem(path: Path) -> str:
    return path.stem


def check_split(root: Path, split):
    frame_root, gt_root = paired_roots(root, split)
    split_name = "/".join(split)
    if not frame_root.exists() or not gt_root.exists():
        print(f"[missing] {split_name}: expected {frame_root} and {gt_root}")
        return False

    ok = True
    videos = sorted(p for p in frame_root.iterdir() if p.is_dir())
    gt_videos = {p.name for p in gt_root.iterdir() if p.is_dir()}
    missing_gt_videos = [p.name for p in videos if p.name not in gt_videos]
    if missing_gt_videos:
        ok = False
        print(f"[video mismatch] {split_name}: {len(missing_gt_videos)} Frame videos lack GT")

    total_frames = 0
    total_missing_masks = 0
    for video_dir in videos:
        gt_dir = gt_root / video_dir.name
        if not gt_dir.exists():
            continue
        frames = sorted(video_dir.glob("*.jpg"))
        masks = {frame_stem(p) for p in gt_dir.glob("*.png")}
        missing = [p.name for p in frames if frame_stem(p) not in masks]
        total_frames += len(frames)
        total_missing_masks += len(missing)
        if missing[:5]:
            ok = False
            preview = ", ".join(missing[:5])
            print(f"[mask mismatch] {split_name}/{video_dir.name}: missing {preview}")

    print(
        f"[checked] {split_name}: {len(videos)} videos, "
        f"{total_frames} frames, {total_missing_masks} missing masks"
    )
    return ok and total_missing_masks == 0


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--root",
        default="../data/SUN-SEG",
        help="Path to the SUN-SEG directory that contains TrainDataset/TestEasyDataset/TestHardDataset.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    results = [check_split(root, split) for split in SPLITS]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
