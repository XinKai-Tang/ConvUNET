import torch
import numpy as np

from typing import Union, Tuple
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, functional as F


class DiceLoss(nn.Module):
    ''' Dice Loss for Segmentation Tasks'''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 square_xy: bool = True,
                 include_bg: bool = True,
                 reduction: str = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `square_xy`: whether using squared result and label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `reduction`: reduction function of dice loss.
        '''
        super(DiceLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(
                "`reduction` of dice loss should be 'mean' or 'sum'!"
            )

        self.n_classes = n_classes
        self.sigmoid_x = sigmoid_x
        self.softmax_x = softmax_x
        self.onehot_y = onehot_y
        self.square_xy = square_xy
        self.include_bg = include_bg
        self.reduction = reduction
        if isinstance(smooth, (tuple, list)):
            self.smooth = smooth
        else:
            self.smooth = (smooth, smooth)

    def forward(self, pred: Tensor, mask: Tensor):
        (sm_nr, sm_dr) = self.smooth

        if self.sigmoid_x:
            pred = torch.sigmoid(pred)
        if self.n_classes > 1:
            if self.softmax_x and self.n_classes == pred.size(1):
                pred = torch.softmax(pred, dim=1)
            if self.onehot_y:
                mask = mask if mask.ndim < 5 else mask.squeeze(dim=1)
                mask = F.one_hot(mask.long(), self.n_classes)
                mask = mask.permute(0, 4, 1, 2, 3).float()
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
        if self.square_xy:
            pred, mask = torch.pow(pred, 2), torch.pow(mask, 2)
        pred_sum = torch.sum(pred, dim=reduce_dims)
        mask_sum = torch.sum(mask, dim=reduce_dims)
        loss = 1. - (2 * insersect + sm_nr) / (pred_sum + mask_sum + sm_dr)

        if self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


class DiceCELoss(nn.Module):
    ''' Dice Loss with Cross Entropy Loss '''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 square_xy: bool = True,
                 include_bg: bool = True,
                 dice_reduct: str = "mean",
                 ce_weight: Tensor = None,
                 ce_reduct: str = "mean",
                 dice_lambda: float = 1.0,
                 ce_lambda: float = 1.0):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `square_xy`: whether using squared result and label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `dice_reduct`: reduction function of dice loss.
        * `ce_weight`: weight of cross entropy loss.
        * `ce_reduct`: reduction function of cross entropy loss.
        * `dice_lambda`: weight coef of dice loss in total loss.
        * `ce_lambda`: weight coef of cross entropy loss in total loss.
        '''
        super(DiceCELoss, self).__init__()
        if dice_lambda < 0:
            raise ValueError(
                f"`dice_lambda` should be no less than 0, but got {dice_lambda}."
            )
        if ce_lambda < 0:
            raise ValueError(
                f"`ce_lambda` should be no less than 0, but got {ce_lambda}."
            )

        self.dice_lambda = dice_lambda
        self.ce_lambda = ce_lambda
        self.dice_loss = DiceLoss(n_classes=n_classes,
                                  smooth=smooth,
                                  sigmoid_x=sigmoid_x,
                                  softmax_x=softmax_x,
                                  onehot_y=onehot_y,
                                  square_xy=square_xy,
                                  include_bg=include_bg,
                                  reduction=dice_reduct)
        self.ce_loss = CrossEntropyLoss(weight=ce_weight,
                                        reduction=ce_reduct)

    def cross_entropy(self, pred: Tensor, mask: Tensor):
        # reducing the channel dimension:
        if pred.size(1) == mask.size(1):
            mask = mask.argmax(dim=1)   # one-hot format
        else:
            mask = mask.squeeze(dim=1)  # (B,C,H,W,D) format
        return self.ce_loss(pred, mask.long())

    def forward(self, pred: Tensor, mask: Tensor):
        dice_loss = self.dice_loss(pred, mask) * self.dice_lambda
        ce_loss = self.cross_entropy(pred, mask) * self.ce_lambda
        return (dice_loss + ce_loss)


class GeneralizedDiceLoss(nn.Module):
    ''' Generalised Dice Loss '''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 include_bg: bool = True,
                 gd_wfunc: str = "square",
                 reduction: str = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `gd_wfunc`: function to transform mask to a GD loss weight factor.
        * `reduction`: reduction function of generalised dice loss.
        '''
        super(GeneralizedDiceLoss, self).__init__()
        if gd_wfunc not in ["square", "simple", "uniform"]:
            raise NotImplementedError(
                "`gd_wfunc` should be 'square', 'simple' or 'uniform'!"
            )
        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(
                "`reduction` of dice loss should be 'mean' or 'sum'!"
            )

        self.n_classes = n_classes
        self.sigmoid_x = sigmoid_x
        self.softmax_x = softmax_x
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

        if self.sigmoid_x:
            pred = torch.sigmoid(pred)
        if self.n_classes > 1:
            if self.softmax_x and self.n_classes == pred.size(1):
                pred = torch.softmax(pred, dim=1)
            if self.onehot_y:
                mask = mask if mask.ndim < 5 else mask.squeeze(dim=1)
                mask = F.one_hot(mask.long(), self.n_classes)
                mask = mask.permute(0, 4, 1, 2, 3).float()
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
        loss = 1.0 - (nr + sm_nr) / (dr + sm_dr)
        if self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


class GeneralizedDiceCELoss(nn.Module):
    ''' Generalized Dice Loss with Cross Entropy Loss '''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 include_bg: bool = True,
                 gd_wfunc: str = "square",
                 gd_reduct: str = "mean",
                 ce_weight: Tensor = None,
                 ce_reduct: str = "mean",
                 gd_lambda: float = 1.0,
                 ce_lambda: float = 1.0):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `gd_wfunc`: function to transform mask to a GD loss weight factor.
        * `gd_reduct`: reduction function of generalized dice loss.
        * `ce_weight`: weight of cross entropy loss.
        * `ce_reduct`: reduction function of cross entropy loss.
        * `gd_lambda`: weight coef of generalized dice loss in total loss.
        * `ce_lambda`: weight coef of cross entropy loss in total loss.
        '''
        super(GeneralizedDiceCELoss, self).__init__()
        if gd_lambda < 0:
            raise ValueError(
                f"`gd_lambda` should be no less than 0, but got {gd_lambda}."
            )
        if ce_lambda < 0:
            raise ValueError(
                f"`ce_lambda` should be no less than 0, but got {ce_lambda}."
            )

        self.gd_lambda = gd_lambda
        self.ce_lambda = ce_lambda
        self.gd_loss = GeneralizedDiceLoss(n_classes=n_classes,
                                           smooth=smooth,
                                           sigmoid_x=sigmoid_x,
                                           softmax_x=softmax_x,
                                           onehot_y=onehot_y,
                                           include_bg=include_bg,
                                           gd_wfunc=gd_wfunc,
                                           reduction=gd_reduct)
        self.ce_loss = CrossEntropyLoss(weight=ce_weight,
                                        reduction=ce_reduct)

    def cross_entropy(self, pred: Tensor, mask: Tensor):
        # reducing the channel dimension:
        if pred.size(1) == mask.size(1):
            mask = mask.argmax(dim=1)   # one-hot format
        else:
            mask = mask.squeeze(dim=1)  # (B,C,H,W,D) format
        return self.ce_loss(pred, mask.long())

    def forward(self, pred: Tensor, mask: Tensor):
        gd_loss = self.gd_loss(pred, mask) * self.gd_lambda
        ce_loss = self.cross_entropy(pred, mask) * self.ce_lambda
        return (gd_loss + ce_loss)


class FocalLoss(nn.Module):
    ''' Focal Loss for Classification Tasks '''

    def __init__(self,
                 n_classes: int = 2,
                 gamma: float = 2,
                 alpha: Variable = None,
                 fl_reduct: str = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `gamma`: decreasing factor.
        * `alpha`: weights of classes.
        * `fl_reduct`: eduction function of focal loss.
        '''
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.fl_reduct = fl_reduct
        self.n_classes = n_classes

        if alpha is None:
            self.alpha = Variable(torch.ones(n_classes, 1))
        elif isinstance(alpha, Variable):
            self.alpha = alpha
        else:
            raise TypeError("The type of `alpha` must be `Variable`.")

    def forward(self, pred: Tensor, label: Tensor):
        if self.alpha.device != pred.device:
            self.alpha = self.alpha.to(pred.device)
        probs = F.softmax(pred, dim=1)
        mask = F.one_hot(label, self.n_classes)
        ids = label.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]
        pt = (probs * mask).sum(1).view(-1, 1)
        loss = -alpha * torch.pow(1-pt, self.gamma) * pt.log()

        if self.fl_reduct == "sum":
            return loss.sum()
        else:
            return loss.mean()

    @staticmethod
    def calc_alpha(class_nums: list):
        ''' calculate `alpha` from numbers of all categories '''
        class_nums = np.array(class_nums, dtype=np.float32)
        if class_nums.ndim != 1:
            raise ValueError("Dimension of `class_nums` should be 1.")
        if any(c <= 0 for c in class_nums):
            raise ValueError("The number of objects of all " +
                             "classes should be greater than 0.")
        class_nums /= class_nums.sum()
        alpha = np.array([[1-c] for c in class_nums], dtype=np.float32)
        alpha = Variable(torch.from_numpy(alpha))
        return alpha
