import torch
import numpy as np

from typing import Union, Tuple
from torch import Tensor, nn
from torch.nn import functional as F


def confusion_matrix(pred: np.ndarray, mask: np.ndarray,
                     n_classes: int, ignore_bg: bool = False):
    ''' compute confusion matrix (for Numpy.Array) '''                
    assert pred.shape == mask.shape
    tp_list, fp_list, tn_list, fn_list = [], [], [], []
    for i in range(int(ignore_bg), n_classes):
        p1 = (pred == i).astype("int16")
        m1 = (mask == i).astype("int16") 
        m0 = (mask != i).astype("int16") 
        tp = (p1 + m1) == 2             # true positive
        tn = (p1 + m1) == 0             # true negetive
        fn = m1 - tp.astype("int16")    # false negetive
        fp = m0 - tn.astype("int16")    # false positive
        tp_list.append(np.sum(np.sum(np.sum(tp))))
        tn_list.append(np.sum(np.sum(np.sum(tn))))
        fn_list.append(np.sum(np.sum(np.sum(fn))))
        fp_list.append(np.sum(np.sum(np.sum(fp))))
    return np.stack([tp_list, fp_list, tn_list, fn_list], axis=-1)


def compute_matrix_metric(method: str, matrix: np.ndarray, 
                          sm: float = 1e-6):
    ''' compute confusion matrix related metric '''
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, ...]
    if matrix.shape[-1] != 4:
        raise ValueError("the last dim of `matrix` should be 4!")
    
    tp = matrix[..., 0].astype("float32")
    fp = matrix[..., 1].astype("float32")
    tn = matrix[..., 2].astype("float32")
    fn = matrix[..., 3].astype("float32")
    p, n = tp + fn, fp + tn
    method = method.lower().replace(" ", "_")
    
    if method in ["tpr", "sensitivity", "recall"]:
        nr, dr = tp, p
    elif method in ["tnr", "specificity"]:
        nr, dr = tn, n
    elif method in ["ppv", "precision"]:
        nr, dr = tp, (tp + fp)
    elif method in ["npv"]:
        nr, dr = tn, (tn + fn)
    elif method in ["fnr", "miss_rate"]:
        nr, dr = fn, p
    elif method in ["fpr", "fall_out"]:
        nr, dr = fp, n
    elif method in ["acc"]:
        nr, dr = (tp + tn), (p + n)
    elif method in ["f1"]:
        nr, dr = tp * 2., (tp * 2. + fn + fp)
    else:
        raise ValueError(f"Method `{method}` is not supported!")
    return (nr + sm) / (dr + sm)


def dice_func(pred: np.ndarray, mask: np.ndarray, n_classes: int, 
              ignore_bg: bool = False, sm: float = 1e-6):
    ''' compute dice (for Numpy.Array) '''
    assert pred.shape == mask.shape
    dsc_list = []
    for i in range(int(ignore_bg), n_classes):
        x, y = pred == i, mask == i
        intersect = np.sum(np.sum(np.sum(x * y)))
        y_sum = np.sum(np.sum(np.sum(y)))
        x_sum = np.sum(np.sum(np.sum(x)))
        dsc = (2 * intersect + sm) / (x_sum + y_sum + sm)
        dsc_list.append(dsc)
    return dsc_list


class DiceMetric(nn.Module):
    ''' Dice Metric of Segmentation Tasks '''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = 1e-6,
                 argmax_x: bool = True,
                 onehot_y: bool = True,
                 include_bg: bool = True,
                 reduction: Union[str, None] = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `argmax_x`: whether using `argmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `reduction`: reduction function of dice metric.
        '''
        super(DiceMetric, self).__init__()
        if reduction not in ["mean", None]:
            raise NotImplementedError(
                "`reduction` of dice should be 'mean' or None!"
            )

        self.n_classes = n_classes
        self.argmax_x = argmax_x
        self.onehot_y = onehot_y
        self.include_bg = include_bg
        self.reduction = reduction
        if isinstance(smooth, (tuple, list)):
            self.smooth = smooth
        else:
            self.smooth = (smooth, smooth)

    def forward(self, pred: Tensor, mask: Tensor):
        dice_list = []
        (sm_nr, sm_dr) = self.smooth

        if self.n_classes > 1:
            if self.argmax_x and self.n_classes == pred.size(1):
                pred = torch.argmax(pred, dim=1)
                pred = F.one_hot(pred.long(), self.n_classes)
                pred = pred.permute(0, 4, 1, 2, 3)
            if self.onehot_y:
                mask = mask if mask.ndim < 5 else mask.squeeze(dim=1)
                mask = F.one_hot(mask.long(), self.n_classes)
                mask = mask.permute(0, 4, 1, 2, 3)
            if not self.include_bg:     # ignore background class
                pred = pred[:, 1:] if pred.size(1) > 1 else pred
                mask = mask[:, 1:] if mask.size(1) > 1 else mask
        if pred.ndim != mask.ndim or pred.size(1) != mask.size(1):
            raise ValueError(
                f"The shape of `pred`({pred.shape}) and " +
                f"`mask`({mask.shape}) should be the same."
            )

        for i in range(mask.size(1)):
            intersect = torch.sum(pred[:, i, ...] * mask[:, i, ...])
            msk_sum = torch.sum(mask[:, i, ...])
            pre_sum = torch.sum(pred[:, i, ...])
            dice = (2 * intersect + sm_nr) / (msk_sum + pre_sum + sm_dr)
            dice_list.append(dice.item())

        if self.reduction == "mean":
            return np.mean(dice_list)
        else:
            return dice_list


class GeneralizedDiceMetric(nn.Module):
    ''' Generalized Dice Metric of Segmentation Tasks '''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = 1e-6,
                 argmax_x: bool = True,
                 onehot_y: bool = True,
                 include_bg: bool = True,
                 gd_wfunc: str = "square",
                 reduction: Union[str, None] = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `argmax_x`: whether using `argmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `gd_wfunc`: function to transform mask to a Generalized Dice weights.
        * `reduction`: reduction function of dice metric.
        '''
        super(GeneralizedDiceMetric, self).__init__()
        if gd_wfunc not in ["square", "simple", "uniform"]:
            raise NotImplementedError(
                "`gd_wfunc` should be 'square', 'simple' or 'uniform'!"
            )
        if reduction not in ["mean", None]:
            raise NotImplementedError(
                "`reduction` of dice should be 'mean' or None!"
            )

        self.n_classes = n_classes
        self.argmax_x = argmax_x
        self.onehot_y = onehot_y
        self.include_bg = include_bg
        self.gd_wfunc = gd_wfunc
        self.reduction = reduction
        if isinstance(smooth, (tuple, list)):
            self.smooth = smooth
        else:
            self.smooth = (smooth, smooth)

    def _weight_func_(self, gt: Tensor):
        if self.gd_wfunc == "simple":
            return torch.reciprocal(gt)
        elif self.gd_wfunc == "square":
            return torch.reciprocal(gt * gt)
        else:
            return torch.ones_like(gt)

    def forward(self, pred: Tensor, mask: Tensor):
        (sm_nr, sm_dr) = self.smooth

        if self.n_classes > 1:
            if self.argmax_x and self.n_classes == pred.size(1):
                pred = torch.argmax(pred, dim=1)
                pred = F.one_hot(pred.long(), self.n_classes)
                pred = pred.permute(0, 4, 1, 2, 3)
            if self.onehot_y:
                mask = mask if mask.ndim < 5 else mask.squeeze(dim=1)
                mask = F.one_hot(mask.long(), self.n_classes)
                mask = mask.permute(0, 4, 1, 2, 3)
            if not self.include_bg:     # ignore background class
                pred = pred[:, 1:] if pred.size(1) > 1 else pred
                mask = mask[:, 1:] if mask.size(1) > 1 else mask
        if pred.ndim != mask.ndim or pred.size(1) != mask.size(1):
            raise ValueError(
                f"The shape of `pred`({pred.shape}) and " +
                f"`mask`({mask.shape}) should be the same."
            )

        # only reducing spatial dimensions:
        reduce_dims = torch.arange(2, pred.ndim).tolist()
        insersect = torch.sum(pred * mask, dim=reduce_dims)
        pred_sum = torch.sum(pred, dim=reduce_dims)
        mask_sum = torch.sum(mask, dim=reduce_dims)

        # computering weights of all classes:
        weight = self._weight_func_(mask_sum.float())
        inf_flags = torch.isinf(weight)
        weight[inf_flags] = 0.0
        max_vals = torch.max(weight, dim=1)[0].unsqueeze(dim=1)
        weight += inf_flags * max_vals

        nr = 2.0 * (weight * insersect).sum(dim=1, keepdim=True)
        dr = weight * (pred_sum + mask_sum).sum(dim=1, keepdim=True)
        dice = (nr + sm_nr) / (dr + sm_dr)
        dice_list = [dice[:, i, ...].item() for i in range(dice.size(1))]

        if self.reduction == "mean":
            return np.mean(dice_list)
        else:
            return dice_list


class AccMetric(nn.Module):
    ''' Accuracy Metric of Classification Tasks '''

    def __init__(self,
                 argmax_x: bool = True,
                 maximum_y: bool = True):
        ''' Args:
        * `argmax_x`: whether using `argmax` to process the result.
        * `maximum_y`: whether using `maximum` to process the label. 
        '''
        super(AccMetric, self).__init__()
        self.argmax_x = argmax_x
        self.maximum_y = maximum_y

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        if self.argmax_x and pred.ndim >= 2:
            pred = pred.argmax(dim=1)
        if self.maximum_y and label.ndim >= 3:
            label = self.get_label(label)
        if pred.ndim != label.ndim and pred.size(0) != label.size(0):
            raise ValueError(
                f"The shape of `pred`({pred.shape}) and " +
                f"`label`({label.shape}) should be the same."
            )
        acc = torch.eq(pred, label).float().mean()
        return acc

    @staticmethod
    def get_label(masks: torch.Tensor):
        ''' get classification labels from segmentation masks '''
        labs = []
        for msk in masks:
            labs.append(torch.max(msk).item())
        labels = torch.tensor(labs, dtype=torch.int64,
                              device=masks.device)
        return labels
