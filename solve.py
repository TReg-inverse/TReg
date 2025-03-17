import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from munch import munchify
from PIL import Image
import torchvision.transforms as transforms

from data.dataloader import get_dataloader
from functions.degradation import get_degradation
from functions.jpeg import build_jpeg
from solver.latent_diffusion import get_solver
from utils.img_util import draw_img
from utils.log_util import Logger

# suppress warning message from CLIP
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def reshape_y(y):
    b = y.shape[0]

    if y.ndim == 4:  # inp or cs
        return y
    try:  # others
        h = w = int(math.sqrt(y.shape[-1]/3))
        y = y.view(b, 3, h, w)
    except:  # colorization
        h = w = int(math.sqrt(y.shape[-1]))
        y = y.view(b, 1, h, w)
    return y

def load_img(img_path: Path, rgb: Optional[bool]=True):
    """
    Load a single image as torch.Tensor.
    Note that the pixel values will be normalized into [0, 1] due to ToTensor().
    Args:
        img_path(Path): image file path (all extensions compatible with PIL.Image.open)
        rgb(Optional(bool)): True for RGB image and False for Gray image.
    """
    mode = 'RGB' if rgb else 'L'
    img = Image.open(img_path).convert(mode)
    tf = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor()])
    img = tf(img)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    return img

def prepare_workdir(root_dir: Path):
    """
    Create work-directory and sub-directories.
    Args:
        root_dir(Path): root work directory
    """
    root_dir.mkdir(parents=True, exist_ok=True)
    prefixs = ['input', 'label', 'recon', 'ypred']
    for prefix in prefixs:
        root_dir.joinpath(prefix).mkdir(exist_ok=True)

def load_config(config_path: Path):
    """
    Load .yaml file as dictionary and convert to namespace.
    Args:
        config_path(Path): configuration file path
    Returns:
        Munch: Loaded configuration namespace
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return munchify(config)

def get_loader(use_dataloader: bool,
                   dataset: str,
                   root: Path,
                   img_path: Path=None,
                   **kwargs):

    log = kwargs.get('log')
    if log is not None:
        if use_dataloader:
            log.info(f"Dataset '{dataset}' is loaded from {root}.")
        else:
            log.info(f"Image is loaded from {img_path}.")

    # multiple images case
    if use_dataloader:
        loader = get_dataloader(dataset=dataset,
                                root=root,
                                batch_size=1,
                                num_workers=1,
                                train=False,
                                rescaled=False,
                                **kwargs)
    # single image case
    else: 
        loader = [load_img(img_path)]
    return loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--solver_config', type=Path, default='./configs/solver/treg.yaml')
    parser.add_argument('--task_config', type=Path, default='./configs/task/super-resolution.yaml')
    # For text prompt
    parser.add_argument('--null_prompt', type=str, default="")
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--cfg_guidance', type=float, default=4.0)
    parser.add_argument('--cg_lamb', type=float, default=1e-4)
    parser.add_argument('--null_lr', type=float, default=8e-4)
    # For log
    parser.add_argument('--workdir', type=Path, default='workdir')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    # For data
    parser.add_argument('--use_dataloader', action='store_true', default=False)
    parser.add_argument('--use_DPS', action='store_true', default=False)
    parser.add_argument('--use_AN', action='store_true', default=False)
    parser.add_argument('--dps_lamb', type=float, default=None)
    parser.add_argument('--dataset', type=str, help="Should be given when --use_dataloader=True")
    parser.add_argument('--root', type=Path, help="Should be gven when --use_dataloader=True")
    parser.add_argument('--img_path', type=Path, default='.')
    parser.add_argument('--target', type=str)

    args = parser.parse_args()
    log = Logger().initLogger()

    log.info(f"Create working directory: {args.workdir}.")
    prepare_workdir(args.workdir)
    log.info(f"Device: {args.device}.")
    log.info(f"Random seed set to {args.seed}.")
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load solver / task config
    log.info('Load configurations.')
    task_conf = load_config(args.task_config)
    solver_conf = load_config(args.solver_config)

    # load solver
    solver = get_solver(name=solver_conf.name,
                        device=args.device,
                        solver_config=solver_conf.params)

    # prepare dataloader
    loader = get_loader(use_dataloader=args.use_dataloader,
                        dataset=args.dataset,
                        root=args.root,
                        img_path=args.img_path,
                        log=log,
                        target=args.target)

    # load operator
    operator = get_degradation(name=task_conf.name,
                               device=args.device,
                               deg_config=task_conf.params) 

    # run solver
    for i, img in enumerate(loader):
        log.info(f"Solve inverse problem for image {i+1}/{len(loader)}.")

        # prepare measurement
        img = img.to(args.device)
        img = (img - 0.5) / 0.5
        y = operator.A(img)
        noise = torch.randn_like(y).to(args.device)
        y = y + 0.01 * noise

        fname = str(i).zfill(5) + f'-{args.prompt}.png'
        draw_img(reshape_y(operator.At(y)), args.workdir.joinpath('input', fname))
        solution = solver(measurement=y,
                          operator=operator,
                          cfg_guidance=args.cfg_guidance,
                          prompt=[args.null_prompt, args.prompt],
                          use_DPS=args.use_DPS,
                          use_AN=args.use_AN,
                          cg_lamb=args.cg_lamb,
                          dps_lamb=args.dps_lamb,
                          workdir=args.workdir,
                          null_lr=args.null_lr,
                          log=log)

        draw_img(img * 2 + 0.5, args.workdir.joinpath('label', fname))
        draw_img(solution, args.workdir.joinpath('recon', fname))
        draw_img(reshape_y(operator.At(operator.A(solution.to(args.device)))), args.workdir.joinpath('ypred', fname))


if __name__ == '__main__':
    main()
