import time
import torch
import numpy as np

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader
from utils.metrics import dice_func, compute_matrix_metric, confusion_matrix


def test():
    ep_time = time.time()
    dsc_list, ppv_list, time_list = [], [], []
    tpr_list, tnr_list, fnr_list, fpr_list = [], [], [], []
    test_loader = get_loader(args, is_test=True)
    seg_model = get_seg_model(args, pretrained=True)
    seg_model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        # GPU warm-up
        img = torch.randn(size=(1, 1, args.roi_z, args.roi_y, args.roi_x),
                          dtype=torch.float32, device=args.device)
        for _ in range(4): _ = seg_model(img)
        # start test model
        for bid, (img, msk, _) in enumerate(test_loader):
            starter.record()
            pred = sliding_window_infer(seg_model, img, args)
            ender.record()
            # waiting for GPU sync
            torch.cuda.synchronize()
            time_list.append(starter.elapsed_time(ender) / 1000)
            # postprocessing
            pred = pred.cpu().numpy()
            mask = msk.squeeze(0).squeeze(0).cpu().numpy()
            # computing metrics
            matrix = confusion_matrix(pred, mask, args.out_channels, ignore_bg=False)
            ppvs = compute_matrix_metric("ppv", matrix)     # Positive Predictive Value
            tprs = compute_matrix_metric("tpr", matrix)     # True Positive Rate
            tnrs = compute_matrix_metric("tnr", matrix)     # True Negative Rate
            fprs = compute_matrix_metric("fpr", matrix)     # False Positive Rate
            fnrs = compute_matrix_metric("fnr", matrix)     # False Negative Rate
            dscs = dice_func(pred, mask, args.out_channels, ignore_bg=False)
            # add into list
            ppv_list.append(np.mean(ppvs))
            tpr_list.append(np.mean(tprs))
            tnr_list.append(np.mean(tnrs))
            fpr_list.append(np.mean(fprs))
            fnr_list.append(np.mean(fnrs))
            dsc_list.append(np.mean(dscs))
            print("【Test】Step: %d, DSC: %.4f," % (bid, dsc_list[-1]),
                  "tpr: %.4f, tnr: %.4f," % (tpr_list[-1], tnr_list[-1]),
                  "fpr: %.4f, fnr: %.4f," % (fpr_list[-1], fnr_list[-1]),
                  "PPV: %.4f, Time: %.2fs" % (ppv_list[-1], time_list[-1]))
    # compute throughput
    throughput = 1.0 / np.mean(time_list) if len(time_list) > 0 else 0
    print("【Test】Throughput: %.2f img/s, avg DSC: %.4f," % (throughput, np.mean(dsc_list)),
          "avg tpr: %.4f, avg tnr: %.4f," % (np.mean(tpr_list), np.mean(tnr_list)),
          "avg fpr: %.4f, avg fnr: %.4f," % (np.mean(fpr_list), np.mean(fnr_list)),
          "avg PPV: %.4f, Time: %.2fs" % (np.mean(ppv_list), time.time() - ep_time))


if __name__ == "__main__":
    test()
