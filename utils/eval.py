import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics

from .print_log import train_log


def eval_net(net, val_loader, device):
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
        return tot / n_val
    else:
        # print('Validation pred values:', pred_ori_list, '\nValidation true values:', true_list)
        train_log.info('\n'+metrics.classification_report(true_list, pred_list))
        return tot / n_val