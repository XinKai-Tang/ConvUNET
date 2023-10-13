import time
import torch
import numpy as np

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader
from utils.losses import DiceCELoss
from utils.metrics import DiceMetric, dice_func


def cross_validate():
    val_writer = LogWriter(args.log_save_dir, prefix=args.seg_model)
    loss_func = DiceCELoss(n_classes=args.out_channels,
                           dice_lambda=1, ce_lambda=1)
    acc_func = DiceMetric(n_classes=args.out_channels)

    all_time = time.time()
    max_dice, num_images = [], []
    for fd in range(args.num_folds):
        args.fold = fd      # modify `fold`
        tra_loader, val_loader = get_loader(args, is_test=False)
        seg_model = get_seg_model(args, pretrained=False)
        optim, scheduler = get_optim(seg_model, args)

        fd_time = time.time()
        max_dice.append(0)
        num_images.append(len(val_loader))
        for ep in range(args.max_epochs):
            loss = run_epoch(fold=fd, epoch=ep, seg_model=seg_model,
                             loader=tra_loader, optim=optim, loss_func=loss_func)
            if scheduler is not None:
                scheduler.step()
            if (ep + 1) % args.val_freq == 0:
                acc = val_epoch(fold=fd, epoch=ep, seg_model=seg_model,
                                loader=val_loader, acc_func=acc_func)
                val_writer.add_row({
                    "fold": fd,
                    "epoch": ep,
                    "train_loss": loss,
                    "val_dice": acc,
                })
                val_writer.save()
                if acc > max_dice[-1]:
                    print("【Val】New Best Acc: %f -> %f" % (max_dice[-1], acc))
                    max_dice[-1] = acc
                    save_ckpts(seg_model, optim, args,
                               save_name=f"best_model_f{fd}.pth")
            save_ckpts(seg_model, optim, args, save_name="latest_model.pth")
        print("【Cross Val】Fold: %d, Best Acc: %f," % (args.fold, max_dice[-1]),
              "Time: %.2fh" % ((time.time() - fd_time) / 3600))
        del tra_loader, val_loader, seg_model, optim, scheduler
        torch.cuda.empty_cache()

    avg_dice = np.multiply(np.array(num_images), np.array(max_dice))
    avg_dice = np.sum(avg_dice) / np.sum(num_images)
    print("【Best Acc】", [round(a, 4) for a in max_dice])
    print("【FINISHED】Mean Acc: %f, " % (avg_dice),
          "Time: %.2fh" % ((time.time() - all_time) / 3600))


def run_epoch(fold, epoch, seg_model, loader, optim, loss_func):
    seg_model.train()
    ep_time = time.time()
    loss_list = []
    for bid, (img, msk, _) in enumerate(loader):
        start = time.time()
        out = seg_model(img)
        loss = loss_func(out.to(args.device), msk.to(args.device))
        loss_list.append(loss.item())
        print("【Train】Fold: %d, Epoch: %d," % (fold, epoch),
              "Batch: %d, Loss: %f," % (bid, loss.item()),
              "Time: %.2fs" % (time.time() - start))
        optim.zero_grad()
        loss.backward()
        optim.step()
    avg_loss = np.mean(loss_list)
    print("【Train】Fold: %d, Epoch: %d," % (fold, epoch),
          "Mean Loss: %f, Time: %.2fs" % (avg_loss, time.time() - ep_time))
    return avg_loss


def val_epoch(fold, epoch, seg_model, loader, acc_func):
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
            print("【Val】Fold: %d, Epoch: %d," % (fold, epoch),
                  "Batch: %d, Dice: %f," % (bid, acc_list[-1]),
                  "Time: %.2fs" % (time.time() - start))
    avg_acc = np.mean(acc_list)
    print("【Val】Fold: %d, Epoch: %d," % (fold, epoch),
          "Mean Dice: %f, Time: %.2fs" % (avg_acc, time.time() - ep_time))
    return avg_acc


if __name__ == "__main__":
    cross_validate()
