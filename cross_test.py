import time
import torch
import numpy as np

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader
from utils.metrics import dice_func, compute_matrix_metric, confusion_matrix


def test_models():
    val_writer = LogWriter(args.log_save_dir, prefix=args.seg_model)
    tpr_list, tnr_list, fpr_list, fnr_list = [], [], [], []
    dsc_list, ppv_list = [], []
    
    all_time = time.time()
    for fd in range(args.num_folds):
        # modify `fold` and `trainded_model`
        args.fold = fd
        args.trained_model = f"best_model_f{fd}.pth"
        # get dataloader and pretrained model
        train_loader, val_loader = get_loader(args, is_test=False)
        try:
            seg_model = get_seg_model(args, pretrained=True)
        except:
            print(f"Warning: {args.trained_model} not found!")
            continue
        # compute metrics
        rets = test(seg_model, val_loader, val_writer, args)
        dsc_list.extend(rets[0])        
        ppv_list.extend(rets[1])
        tpr_list.extend(rets[2])
        tnr_list.extend(rets[3])
        fpr_list.extend(rets[4])
        fnr_list.extend(rets[5])
        val_writer.save()
        del train_loader, val_loader, seg_model
        torch.cuda.empty_cache()

    # compute mean metrics
    avg_dsc = round(np.mean(dsc_list), 4)
    avg_tpr = round(np.mean(tpr_list), 4)
    avg_tnr = round(np.mean(tnr_list), 4)
    avg_fpr = round(np.mean(fpr_list), 4)
    avg_fnr = round(np.mean(fnr_list), 4)
    avg_ppv = round(np.mean(ppv_list), 4)
    print("【FINISHED】Model: %s," % (args.seg_model),
          "avg DSC: %.4f, avg PPV: %.4f," % (avg_dsc, avg_ppv),
          "avg tpr: %.4f, avg tnr: %.4f," % (avg_tpr, avg_tnr),
          "avg fpr: %.4f, avg fnr: %.4f," % (avg_fpr, avg_fnr),
          "Time: %.2fh" % ((time.time() - all_time) / 3600))

def test(model, loader, writer, args):
    ep_time = time.time()
    dsc_list, ppv_list, time_list = [], [], []
    tpr_list, tnr_list, fnr_list, fpr_list = [], [], [], []

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    model.eval()
    with torch.no_grad():
        # GPU warm-up
        img = torch.randn(size=(1, 1, args.roi_z, args.roi_y, args.roi_x),
                          dtype=torch.float32, device=args.device)
        for _ in range(4): _ = model(img)
        # start test model
        for bid, (img, msk, info) in enumerate(loader):
            starter.record()
            pred = sliding_window_infer(model, img, args)
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
            # add into logs
            writer.add_row({
                "fold": args.fold,
                "fname": info["fname"][0],
                "DSC": round(dsc_list[-1], 5),
                "PPV": round(ppv_list[-1], 5),
                "tpr": round(tpr_list[-1], 5),
                "tnr": round(tnr_list[-1], 5),
                "fpr": round(fpr_list[-1], 5),
                "fnr": round(fnr_list[-1], 5),
            })
            print("【Test】Fold: %d, Step: %d," % (args.fold, bid),
                  "DSC: %.4f, PPV: %.4f," % (dsc_list[-1], ppv_list[-1]),
                  "tpr: %.4f, tnr: %.4f," % (tpr_list[-1], tnr_list[-1]),
                  "fpr: %.4f, fnr: %.4f," % (fpr_list[-1], fnr_list[-1]),
                  "Time: %.2fs" % (time_list[-1]))
    # compute throughput
    throughput = 1.0 / np.mean(time_list) if len(time_list) > 0 else 0
    # compute mean metrics
    avg_dsc = round(np.mean(dsc_list), 4)
    avg_tpr = round(np.mean(tpr_list), 4)
    avg_tnr = round(np.mean(tnr_list), 4)
    avg_fpr = round(np.mean(fpr_list), 4)
    avg_fnr = round(np.mean(fnr_list), 4)
    avg_ppv = round(np.mean(ppv_list), 4)
    print("【Test】Fold: %d, Throughput: %.2f img/s," % (args.fold, throughput),
          "avg DSC: %.4f, avg PPV: %.4f," % (avg_dsc, avg_ppv),
          "avg tpr: %.4f, avg tnr: %.4f," % (avg_tpr, avg_tnr),
          "avg fpr: %.4f, avg fnr: %.4f," % (avg_fpr, avg_fnr),
          "Time: %.2fs" % (time.time() - ep_time))
    return (dsc_list, ppv_list, tpr_list, tnr_list, fpr_list, fnr_list)


if __name__ == "__main__":
    test_models()
