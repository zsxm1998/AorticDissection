import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.linalg as LA
from tqdm import tqdm
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE


def eval_net(net, val_loader, n_val, device, final=False, PR_curve_save_dir=None):
    train_log = logging.getLogger('train_log')
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    net.eval()
    category_type = torch.float32 if module.n_classes == 1 else torch.long
    num_val_batches = len(val_loader)  # the number of batch
    tot_loss = 0

    true_list = []
    pred_list = []
    pred_ori_list = []
    for imgs, true_categories in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
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
            tot_loss += F.cross_entropy(categories_pred, true_categories, reduction='sum').item()
        else:
            pred = torch.sigmoid(categories_pred)
            pred_ori_list += pred.squeeze(1).tolist()
            pred = (pred > 0.5).float()
            # tot += metrics.f1_score(true_categories.cpu().numpy(), pred.squeeze(-1).cpu().numpy())
            true_list += true_categories.tolist()
            pred_list.extend(pred.squeeze(-1).tolist())
            tot_loss += F.binary_cross_entropy_with_logits(categories_pred, true_categories.unsqueeze(1), reduction='sum').item()
    
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
        train_log.info('Validation report:\n'+metrics.classification_report(true_list, pred_list, digits=4))
        return ( float(np.mean(AP)), tot_loss / n_val ) if not final \
            else ( float(np.mean(AP)), tot_loss / n_val, os.path.join(PR_curve_save_dir, 'PR-curve.png') )
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
        train_log.info('Validation report:\n'+metrics.classification_report(true_list, pred_list, target_names=['negative', 'positive'], digits=4))
        return ( metrics.roc_auc_score(true_list, pred_ori_list), tot_loss / n_val ) if not final \
            else ( metrics.roc_auc_score(true_list, pred_ori_list), tot_loss / n_val, os.path.join(PR_curve_save_dir, 'PR-curve.png') ) #return tot / n_val if not final else os.path.join(PR_curve_save_dir, 'PR-curve.png')



def eval_supcon(net, val_loader, n_val, device, n_classes, final=False, TSNE_save_dir=None):
    train_log = logging.getLogger('train_log')
    net.eval()
    num_val_batches = len(val_loader)  # the number of batch

    true_list = []
    r_list = []
    for imgs, true_categories in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_categories = true_categories.to(device=device, dtype=torch.long)
        true_list.append(true_categories)

        with torch.no_grad():
            _, r = net(imgs)
        r_list.append(r)

    labels = torch.cat(true_list, dim=0)
    reprs = torch.cat(r_list, dim=0)

    n_classes = 2 if n_classes == 1 else n_classes
    mean_list = []
    inner_dis = 0
    for k in range(n_classes):
        reprs_k = reprs[labels == k]
        reprs_k_mean = torch.mean(reprs_k, dim=0, keepdim=True)
        mean_list.append(reprs_k_mean)
        inner_dis += torch.mean(LA.norm(reprs_k-reprs_k_mean, dim=1)).item()
    inner_dis /= n_classes
    mean_matrix = torch.cat(mean_list, dim=0)
    outer_dis = torch.mean(F.pdist(mean_matrix)).item()
    
    net.train()
    if final:
        labels = labels.cpu().numpy()
        reprs = reprs.cpu().numpy()
        reprs_tsne = TSNE(n_jobs=8).fit_transform(reprs)
        vis_x = reprs_tsne[:, 0]
        vis_y = reprs_tsne[:, 1]
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", n_classes), marker='.')
        plt.colorbar(ticks=range(n_classes))
        plt.savefig(os.path.join(TSNE_save_dir, 'TSNE.png'))
        return inner_dis / outer_dis, os.path.join(TSNE_save_dir, 'TSNE.png')

    return inner_dis / outer_dis


def eval_combine(supcon, fc, val_loader, n_val, device, n_classes, final=False, PR_curve_save_dir=None, TSNE_save_dir=None):
    train_log = logging.getLogger('train_log')
    supcon.eval()
    fc.eval()
    category_type = torch.float32 if fc.out_features == 1 else torch.long
    num_val_batches = len(val_loader)  # the number of batch
    tot_loss = 0

    true_list = []
    true_tensor_list = []
    pred_list = []
    pred_ori_list = []
    r_list = []
    for imgs, true_categories in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_categories = true_categories.to(device=device, dtype=category_type)
        true_tensor_list.append(true_categories)

        with torch.no_grad():
            _, r = supcon(imgs)
            categories_pred = fc(r)
        r_list.append(r)

        if fc.out_features > 1:
            pred = torch.softmax(categories_pred, dim=1)
            pred_ori_list += pred.tolist()
            pred = pred.argmax(dim=1)
            true_list += true_categories.tolist()
            pred_list.extend(pred.tolist())
            tot_loss += F.cross_entropy(categories_pred, true_categories, reduction='sum').item()
        else:
            pred = torch.sigmoid(categories_pred)
            pred_ori_list += pred.squeeze(1).tolist()
            pred = (pred > 0.5).float()
            true_list += true_categories.tolist()
            pred_list.extend(pred.squeeze(-1).tolist())
            tot_loss += F.binary_cross_entropy_with_logits(categories_pred, true_categories.unsqueeze(1), reduction='sum').item()

    labels = torch.cat(true_tensor_list, dim=0).to(dtype = torch.long)
    reprs = torch.cat(r_list, dim=0)

    n_classes = 2 if n_classes == 1 else n_classes
    mean_list = []
    inner_dis = 0
    for k in range(n_classes):
        reprs_k = reprs[labels == k]
        reprs_k_mean = torch.mean(reprs_k, dim=0, keepdim=True)
        mean_list.append(reprs_k_mean)
        inner_dis += torch.mean(LA.norm(reprs_k-reprs_k_mean, dim=1)).item()
    inner_dis /= n_classes
    mean_matrix = torch.cat(mean_list, dim=0)
    outer_dis = torch.mean(F.pdist(mean_matrix)).item()
    
    supcon.train()
    fc.train()
    if final:
        labels = labels.cpu().numpy()
        reprs = reprs.cpu().numpy()
        reprs_tsne = TSNE(n_jobs=8).fit_transform(reprs)
        vis_x = reprs_tsne[:, 0]
        vis_y = reprs_tsne[:, 1]
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", n_classes), marker='.')
        plt.colorbar(ticks=range(n_classes))
        plt.savefig(os.path.join(TSNE_save_dir, 'TSNE.png'))
    if fc.out_features > 1:
        AP = []
        if final:
            plt.figure("P-R Curve")
            plt.title('Precision/Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        for c in range(fc.out_features):
            c_true_list = [int(item==c) for item in true_list]
            c_pred_ori_list = [item[c] for item in pred_ori_list]
            AP.append(metrics.average_precision_score(c_true_list, c_pred_ori_list))
            if final:
                c_precision, c_recall, _ = metrics.precision_recall_curve(c_true_list, c_pred_ori_list)
                plt.plot(c_recall, c_precision, label=f'class {c}')
        if final:
            plt.ylim(bottom=0)
            plt.legend()
            plt.savefig(os.path.join(PR_curve_save_dir, 'PR-curve.png'))
        train_log.info('Validation report:\n'+metrics.classification_report(true_list, pred_list, digits=4))
        return ( float(np.mean(AP)), tot_loss / n_val, inner_dis / outer_dis ) if not final \
            else ( float(np.mean(AP)), tot_loss / n_val, inner_dis / outer_dis, os.path.join(PR_curve_save_dir, 'PR-curve.png'), os.path.join(TSNE_save_dir, 'TSNE.png') )
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
            plt.legend()
            plt.savefig(os.path.join(PR_curve_save_dir, 'PR-curve.png'))
        train_log.info('Validation report:\n'+metrics.classification_report(true_list, pred_list, target_names=['negative', 'positive'], digits=4))
        return ( metrics.roc_auc_score(true_list, pred_ori_list), tot_loss / n_val, inner_dis / outer_dis ) if not final \
            else ( metrics.roc_auc_score(true_list, pred_ori_list), tot_loss / n_val, inner_dis / outer_dis, os.path.join(PR_curve_save_dir, 'PR-curve.png'), os.path.join(TSNE_save_dir, 'TSNE.png') )