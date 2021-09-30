import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt

from .print_log import train_log


def eval_net(net, val_loader, device, final=False, PR_curve_save_dir=None):
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    net.eval()
    category_type = torch.float32 if module.n_classes == 1 else torch.long
    n_val = len(val_loader)  # the number of batch
    tot = 0

    true_list = []
    pred_list = []
    pred_ori_list = []
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for imgs, true_categories in val_loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_categories = true_categories.to(device=device, dtype=category_type)

            with torch.no_grad():
                categories_pred = net(imgs)

            if module.n_classes > 1:
                pred = torch.softmax(categories_pred, dim=1)
                pred_ori_list += pred.tolist()
                pred = pred.argmax(dim=1)
                true_list += true_categories.tolist()
                pred_list.extend(pred.tolist())
                tot += F.cross_entropy(categories_pred, true_categories).item()
            else:
                pred = torch.sigmoid(categories_pred)
                pred_ori_list += pred.squeeze(1).tolist()
                pred = (pred > 0.5).float()
                # tot += metrics.f1_score(true_categories.cpu().numpy(), pred.squeeze(-1).cpu().numpy())
                true_list += true_categories.tolist()
                pred_list.extend(pred.squeeze(-1).tolist())
                tot += F.binary_cross_entropy_with_logits(categories_pred, true_categories.unsqueeze(1)).item()
            pbar.update()
    
    net.train()
    if module.n_classes > 1:
        AP = []
        if final:
            plt.figure("P-R Curve")
            plt.title('Precision/Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        for c in range(module.n_classes):
            c_true_list = [int(item==c) for item in true_list]
            c_pred_ori_list = [item[c] for item in pred_ori_list]
            AP.append(metrics.average_precision_score(c_true_list, c_pred_ori_list))
            if final:
                c_precision, c_recall, _ = metrics.precision_recall_curve(c_true_list, c_pred_ori_list)
                plt.plot(c_recall, c_precision, label=f'class {c}')
        if final:
            plt.ylim(bottom=0)
            plt.legend() #plt.legend(loc="lower left")
            plt.savefig(os.path.join(PR_curve_save_dir, 'PR-curve.png'))
        train_log.info('\n'+metrics.classification_report(true_list, pred_list))
        return ( float(np.mean(AP)), tot / n_val ) if not final \
            else ( float(np.mean(AP)), tot / n_val, os.path.join(PR_curve_save_dir, 'PR-curve.png') )
    else:
        if final:
            precision1, recall1, _ = metrics.precision_recall_curve(true_list, pred_ori_list)
            precision0, recall0, _ = metrics.precision_recall_curve(list(map(lambda x: 1-x, true_list)), list(map(lambda x: 1-x, pred_ori_list)))
            plt.figure("P-R Curve")
            plt.title('Precision/Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.plot(recall0, precision0, label='negative')
            plt.plot(recall1, precision1, label='positive')
            plt.ylim(bottom=0)
            plt.legend() #plt.legend(loc="lower left")
            plt.savefig(os.path.join(PR_curve_save_dir, 'PR-curve.png'))
        # print('Validation pred values:', pred_ori_list, '\nValidation true values:', true_list)
        train_log.info('\n'+metrics.classification_report(true_list, pred_list, target_names=['negative', 'positive']))
        return ( metrics.roc_auc_score(true_list, pred_ori_list), tot / n_val ) if not final \
            else ( metrics.roc_auc_score(true_list, pred_ori_list), tot / n_val, os.path.join(PR_curve_save_dir, 'PR-curve.png') ) #return tot / n_val if not final else os.path.join(PR_curve_save_dir, 'PR-curve.png')
