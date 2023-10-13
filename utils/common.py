import os
import time
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk

from argparse import ArgumentParser
from typing import Union

from torch.nn import Module, DataParallel, functional as F
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.optimizers.lr_scheduler import WarmupCosineSchedule

from nets import UNETR, SwinUNETR, UXNET, SAN, UPerNeXt, UNETR_PP


def get_seg_model(args: ArgumentParser, pretrained: bool = True):
    ''' get segmentation model '''
    seg_model_name = args.seg_model.lower()
    if seg_model_name == "uxnet":
        seg_model = UXNET(in_channels=args.in_channels,
                          out_channels=args.out_channels)
    elif seg_model_name == "san":
        seg_model = SAN(in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        deep_sup=False)
    elif seg_model_name == "upernextv1":
        seg_model = UPerNeXt(in_channels=args.in_channels,
                             out_channels=args.out_channels,
                             use_grn=False)
    elif seg_model_name == "upernextv2":
        seg_model = UPerNeXt(in_channels=args.in_channels,
                             out_channels=args.out_channels,
                             use_grn=True)
    elif seg_model_name == "swinunetr":
        seg_model = SwinUNETR(in_channels=args.in_channels,
                              out_channels=args.out_channels)
    elif seg_model_name == "unetr++":
        seg_model = UNETR_PP(in_channels=args.in_channels,
                             out_channels=args.out_channels,
                             img_size=(args.roi_z, args.roi_x, args.roi_y))
    elif seg_model_name == "unetr":
        seg_model = UNETR(in_channels=args.in_channels,
                          out_channels=args.out_channels,
                          img_size=(args.roi_z, args.roi_x, args.roi_y))
    else:
        raise ValueError(
            f"Segmentation Model `{args.seg_model}` is not supported!"
        )
    seg_model = DataParallel(seg_model).to(args.device)

    if pretrained:
        path = os.path.join(args.model_save_dir,
                            args.seg_model, args.trained_model)
        state_dict = torch.load(path)
        seg_model.load_state_dict(state_dict["net"])
        print("LOAD checkpoints from `%s`..." % path)
    return seg_model


def get_optim(model: Module, args: ArgumentParser):
    ''' get optimizer and learning rate scheduler '''
    optim_name = args.optim_name.lower()
    lrschedule = args.lrschedule.lower()

    if optim_name == "sgd":
        optim = SGD(params=model.parameters(),
                    lr=args.optim_lr,
                    weight_decay=args.reg_weight)
    elif optim_name == "adam":
        optim = Adam(params=model.parameters(),
                     lr=args.optim_lr,
                     weight_decay=args.reg_weight)
    else:
        optim = AdamW(params=model.parameters(),
                      lr=args.optim_lr,
                      weight_decay=args.reg_weight)

    if lrschedule == "cosine":
        scheduler = CosineAnnealingLR(optim, T_max=args.max_epochs)
    elif lrschedule == "warmupcosine":
        scheduler = WarmupCosineSchedule(optim, t_total=args.max_epochs,
                                         warmup_steps=args.warmup_steps)
    else:
        scheduler = None
    return optim, scheduler


def save_ckpts(model: Module,
               optim: Optimizer,
               args: ArgumentParser,
               save_name: str):
    ''' save checkpoints (model and optimizer states)  '''
    state_dict = {
        "net": model.state_dict(),
        "optimizer": optim.state_dict()
    }
    path = os.path.join(args.model_save_dir, args.seg_model)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, save_name)
    torch.save(state_dict, path)
    print("SAVE checkpoints to `%s`..." % path)


def save_segmentation(result: Union[np.ndarray, torch.Tensor],
                      args: ArgumentParser,
                      axis: int = 1,
                      save_name: str = "segmentation.npy"):
    ''' save segmentation result as a file '''
    # get the save path of the segmentation result:
    path = os.path.join(args.seg_save_dir, args.model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, save_name)
    # post-process the segmentation result:
    if isinstance(result, torch.Tensor):
        result = torch.softmax(result, axis).detach().cpu().numpy()
    if result.shape[axis] > 1:
        result = np.argmax(result, axis).astype(np.uint8)[0]
    # save the quantification result to a file:
    if path[-4:] == ".npy":
        np.save(path, result)
    else:
        res_img = sitk.GetImageFromArray(result)
        sitk.WriteImage(res_img, path)
    print("SAVE segmentation to `%s`..." % path)


def sliding_window_infer(model: Module,
                         image: torch.Tensor,
                         args: ArgumentParser):
    ''' segment images by sliding window '''
    if image.shape[0] != 1:
        raise ValueError("Batch size of image should be 1.")
    if image.ndim != 5:
        raise ValueError("Number of image dimension should be 5.")

    (D, H, W) = image.shape[2:]
    pad_size, win_size = [], [args.roi_z, args.roi_x, args.roi_y]
    for k in range(len(image.shape)-1, 1, -1):
        diff = image.shape[k] % win_size[k-2]
        diff = (win_size[k-2] - diff) if diff > 0 else 0
        pad_size.extend([0, diff])
    # 填充image，使得它们的尺寸恰好是win_size的整数倍：
    image = F.pad(image, pad_size, "constant", image.min().item())
    pred = torch.zeros(image.shape[2:], dtype=torch.float32)

    for x in range(0, image.size(-3), win_size[0]):
        for y in range(0, image.size(-2), win_size[1]):
            for z in range(0, image.size(-1), win_size[2]):
                img = image[:, :, x: x+win_size[0],
                            y: y+win_size[1], z: z+win_size[2]]
                out = model(img)
                out = torch.argmax(out, dim=1).squeeze(dim=0)
                pred[x: x+win_size[0], y: y+win_size[1],
                     z: z+win_size[2]] = out.float()
    pred: torch.Tensor = pred[0: D, 0: H, 0: W].contiguous()
    return pred


class LogWriter:
    ''' Log Writer Based on Pandas '''

    def __init__(self, save_dir: str, prefix: str = None):
        ''' Args:
        * `save_dir`: save place of log files.
        * `prefix`: prefix-name of the log file.
        '''
        self.data = pd.DataFrame()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        now = time.strftime("%y%m%d%H%M", time.localtime())
        fname = f"{prefix}-{now}.csv" if prefix else f"{now}.csv"
        self.path = os.path.join(save_dir, fname)

    def add_row(self, data: dict):
        temp = pd.DataFrame(data, index=[0])
        self.data = pd.concat([self.data, temp], ignore_index=True)

    def save(self):
        self.data.to_csv(self.path, index=False)
        print("SAVE runtime logs to `%s`..." % self.path)
