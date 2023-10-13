import time
import torch
import numpy as np

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader
from utils.losses import DiceCELoss
from utils.metrics import DiceMetric, dice_func


def train():
    tra_loader, val_loader = get_loader(args, is_test=False)
    seg_model = get_seg_model(args, pretrained=False)
    optim, scheduler = get_optim(seg_model, args)
    val_writer = LogWriter(args.log_save_dir, prefix=args.seg_model)
    loss_func = DiceCELoss(n_classes=args.out_channels,
                           dice_lambda=1, ce_lambda=4)
    acc_func = DiceMetric(n_classes=args.out_channels)

    max_dice, all_time = 0, time.time()
    for ep in range(args.max_epochs):
        loss = run_epoch(ep, seg_model, tra_loader, optim, loss_func)
        if scheduler is not None:
            scheduler.step()
        if (ep + 1) % args.val_freq == 0:
            acc = val_epoch(ep, seg_model, val_loader, acc_func)
            val_writer.add_row({
                "epoch": ep,
                "train_loss": loss,
                "val_dice": acc,
            })
            val_writer.save()
            if acc > max_dice:
                print("【Val】New Best Acc: %f -> %f" % (max_dice, acc))
                max_dice = acc
                save_ckpts(seg_model, optim, args, save_name="best_model.pth")
        save_ckpts(seg_model, optim, args, save_name="latest_model.pth")
    print("【FINISHED】Best Acc: %f, " % (max_dice),
          "Time: %.2fh" % ((time.time() - all_time) / 3600))


def run_epoch(epoch, seg_model, loader, optim, loss_func):
    seg_model.train()
    ep_time = time.time()
    loss_list = []
    for bid, (img, msk, _) in enumerate(loader):
        start = time.time()
        out = seg_model(img)
        loss = loss_func(out.to(args.device), msk.to(args.device))
        loss_list.append(loss.item())
        print("【Train】Epoch: %d," % (epoch),
              "Batch: %d, Loss: %f," % (bid, loss.item()),
              "Time: %.2fs" % (time.time() - start))
        optim.zero_grad()
        loss.backward()
        optim.step()
    avg_loss = np.mean(loss_list)
    print("【Train】Epoch: %d," % (epoch),
          "Mean Loss: %f, Time: %.2fs" % (avg_loss, time.time() - ep_time))
    return avg_loss


def val_epoch(epoch, seg_model, loader, acc_func):
    seg_model.eval()
    ep_time = time.time()
    acc_list = []
    with torch.no_grad():
        for bid, (img, msk, _) in enumerate(loader):
            start = time.time()
            pred = sliding_window_infer(seg_model, img, args)
            pred = pred.cpu().numpy()
            mask = msk.squeeze(0).squeeze(0).cpu().numpy()
            accs = dice_func(pred, mask, args.out_channels, ignore_bg=False)
            acc_list.append(np.mean(accs))
            print("【Val】Epoch: %d," % (epoch),
                  "Batch: %d, Dice: %f," % (bid, acc_list[-1]),
                  "Time: %.2fs" % (time.time() - start))
    avg_acc = np.mean(acc_list)
    print("【Val】Epoch: %d," % (epoch),
          "Mean Dice: %f, Time: %.2fs" % (avg_acc, time.time() - ep_time))
    return avg_acc


if __name__ == "__main__":
    train()
