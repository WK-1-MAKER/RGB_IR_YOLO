"""Evaluate an RGB/IR model with RGB pairing preserved or randomly shuffled.

The IR image and labels stay attached to the original sample. Only the RGB
image is borrowed from another sample, which tests whether the model uses
correct RGB/IR correspondence at inference time.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from models.experimental import attempt_load
from test import test
from utils.datasets import LoadImagesAndLabels, LoadMultiModalImagesAndLabels, letterbox
from utils.general import check_dataset, check_file, check_img_size, colorstr, increment_path, set_logging
from utils.torch_utils import select_device


class ShuffledRgbDataset(Dataset):
    """Keep IR/labels fixed while replacing each sample's RGB image."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        if len(dataset.img_files_rgb) != len(dataset.img_files_ir):
            raise ValueError(
                f'RGB/IR image counts differ: {len(dataset.img_files_rgb)} vs '
                f'{len(dataset.img_files_ir)}'
            )
        if len(dataset) < 2:
            raise ValueError('Shuffled RGB evaluation requires at least two samples.')
        rng = np.random.default_rng(seed)
        indices = np.arange(len(dataset))
        self.permutation = rng.permutation(indices)
        while np.any(self.permutation == indices):
            self.permutation = rng.permutation(indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        paired_image, labels, path, shapes = self.dataset[index]
        donor_index = int(self.permutation[index])
        donor_path = self.dataset.img_files_rgb[donor_index]
        donor_rgb = cv2.imread(donor_path)
        if donor_rgb is None:
            raise FileNotFoundError(f'RGB donor image not found: {donor_path}')

        height, width = donor_rgb.shape[:2]
        ratio = self.dataset.img_size / max(height, width)
        if ratio != 1:
            donor_rgb = cv2.resize(
                donor_rgb,
                (int(width * ratio), int(height * ratio)),
                interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR,
            )
        donor_rgb, _, _ = letterbox(
            donor_rgb, paired_image.shape[1:], auto=False, scaleup=False
        )
        donor_rgb = donor_rgb[:, :, ::-1].transpose(2, 0, 1)
        donor_rgb = torch.from_numpy(np.ascontiguousarray(donor_rgb))

        image = torch.cat((donor_rgb, paired_image[3:]), dim=0)
        return image, labels, path, shapes


def collate_fn(batch):
    """Reuse the multimodal dataset's target-index collation."""
    return LoadImagesAndLabels.collate_fn(batch)


def make_dataloader(data, task, imgsz, batch_size, stride, opt, seed=None, workers=4):
    if task == 'test':
        rgb_path = data['test_rgb']
        ir_path = data['test_ir']
    else:
        rgb_path = data['val_rgb']
        ir_path = data['val_ir']

    base = LoadMultiModalImagesAndLabels(
        rgb_path,
        ir_path,
        imgsz,
        batch_size,
        augment=False,
        hyp=None,
        rect=True,
        cache_images=False,
        single_cls=opt.single_cls,
        stride=int(stride),
        pad=0.5,
        prefix=colorstr(f'{task}: '),
    )
    dataset = ShuffledRgbDataset(base, seed) if seed is not None else base
    actual_batch_size = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=actual_batch_size,
        num_workers=min(workers, actual_batch_size if actual_batch_size > 1 else 0),
        pin_memory=True,
        collate_fn=collate_fn,
    )


def evaluate(model, data, task, imgsz, batch_size, stride, opt, save_dir, seed=None):
    dataloader = make_dataloader(
        data, task, imgsz, batch_size, stride, opt, seed=seed, workers=opt.workers
    )
    result, _, _ = test(
        data,
        model=model,
        dataloader=dataloader,
        imgsz=imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        single_cls=opt.single_cls,
        augment=False,
        verbose=opt.verbose,
        save_dir=save_dir,
        plots=False,
        half_precision=not opt.no_half,
    )
    names = ('precision', 'recall', 'map50', 'map75', 'map5095', 'box_loss', 'obj_loss', 'cls_loss')
    return {name: float(value) for name, value in zip(names, result)}


def parse_opt():
    parser = argparse.ArgumentParser(
        description='Evaluate an RGB/IR model with aligned or shuffled RGB pairs.'
    )
    parser.add_argument(
        '--weights',
        default='runs/train/MSDA-FLIR-n-layer_n1-bt20-ep40/weights/last.pt',
        help='model checkpoint path',
    )
    parser.add_argument('--data', default='data.yaml', help='dataset YAML with RGB/IR paths')
    parser.add_argument('--task', choices=('val', 'test'), default='test')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', default='0', help='CUDA device, or cpu')
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--skip-aligned', action='store_true', help='only run shuffled RGB')
    parser.add_argument('--no-half', action='store_true', help='disable FP16 on CUDA')
    parser.add_argument('--project', default='runs/test_shuffle_rgb')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    return parser.parse_args()


def main():
    opt = parse_opt()
    set_logging()
    opt.data = check_file(opt.data)
    with open(opt.data) as stream:
        data = yaml.safe_load(stream)
    check_dataset(data)

    device = select_device(opt.device, batch_size=opt.batch_size)
    model = attempt_load(opt.weights, map_location=device)
    stride = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.img_size, s=stride)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if not opt.skip_aligned:
        print('\n=== aligned RGB/IR ===')
        results['aligned'] = evaluate(
            model, data, opt.task, imgsz, opt.batch_size, stride, opt, save_dir / 'aligned'
        )
        print(json.dumps(results['aligned'], indent=2))

    shuffled = []
    for seed in opt.seeds:
        print(f'\n=== shuffled RGB, seed={seed} ===')
        metrics = evaluate(
            model, data, opt.task, imgsz, opt.batch_size, stride, opt,
            save_dir / f'shuffled_seed{seed}', seed=seed
        )
        shuffled.append({'seed': seed, **metrics})
        print(json.dumps(metrics, indent=2))
    results['shuffled'] = shuffled

    if shuffled:
        for metric in ('map50', 'map5095'):
            values = np.array([row[metric] for row in shuffled], dtype=np.float64)
            print(f'shuffled {metric}: {values.mean():.6f} +/- {values.std():.6f}')

    output = save_dir / 'results.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nResults saved to {output}')


if __name__ == '__main__':
    main()
