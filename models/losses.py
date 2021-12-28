import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        #loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.view(anchor_count, batch_size)
        weight = torch.tensor([12434/6522, 12434/4370, 12424/1149, 12434/393]).to(device)
        batch_weight = torch.gather(weight.unsqueeze(0).repeat(batch_size, 1), 1, labels.to(device)).detach()
        loss = ((loss * batch_weight.T).sum(1) / batch_weight.sum()).mean()

        return loss


class HookModule:
    def __init__(self, model, module):
        self.model = model
        self.handle = module.register_forward_hook(self._get_output)
        
    def _get_output(self, module, inputs, outputs):
        self.outputs = outputs
    
    def grads(self, outputs, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs, inputs=self.outputs, retain_graph=retain_graph, create_graph=create_graph)
        self.model.zero_grad()
        return grads[0]
    
    def remove(self):
        self.handle.remove()


class GradConstraint:

    def __init__(self, model, modules, channel_paths, flag_3d=False):
        self.model = model
        self.modules = modules
        self.hook_modules = []
        self.channels = []
        self.flag_3d = flag_3d

        for channel_path in channel_paths:
            self.channels.append(torch.from_numpy(np.load(channel_path)).cuda())

    def add_hook(self):
        for module in self.modules:
            self.hook_modules.append(HookModule(model=self.model, module=module))

    def remove_hook(self):
        for hook_module in self.hook_modules:
            hook_module.remove()
        self.hook_modules.clear()

    def loss_spatial(self, outputs, labels, masks):
        # nll_loss = torch.nn.NLLLoss()(outputs, labels)
        # grads = self.hook_modules[0].grads(outputs=-nll_loss)
        # masks = transforms.Resize((grads.shape[2], grads.shape[3]))(masks)
        # masks_bg = 1 - masks
        # grads_bg = torch.abs(masks_bg * grads)

        # loss = grads_bg.sum()
        # return loss
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        loss = 0
        for hook_module in self.hook_modules:
            grads = hook_module.grads(outputs=-nll_loss)
            if self.flag_3d:
                masks_bg = F.interpolate(masks, (grads.shape[2], grads.shape[3], grads.shape[4]), mode='trilinear')
            else:
                masks_bg = transforms.Resize((grads.shape[2], grads.shape[3]))(masks)
            masks_bg = 1 - masks_bg
            grads_bg = torch.abs(masks_bg * grads)#F.relu(masks_bg * grads)
            loss += grads_bg.sum()
        return loss

    def loss_channel(self, outputs, labels):
        # high response channel loss
        probs = torch.argsort(-outputs, dim=1)
        labels_ = []
        for i in range(labels.size(0)):
            if probs[i][0] == labels[i]:
                labels_.append(probs[i][1])  # TP rank2
            else:
                labels_.append(probs[i][0])  # FP rank1
        labels_ = torch.tensor(labels_).cuda()
        nll_loss_ = torch.nn.NLLLoss()(outputs, labels_)
        # low response channel loss
        nll_loss = torch.nn.NLLLoss()(outputs, labels)

        loss = 0
        for i, hook_module in enumerate(self.hook_modules):
            # high response channel loss
            loss += self._loss_channel(channels=self.channels[i],
                                  grads=hook_module.grads(outputs=-nll_loss_),
                                  labels=labels_,
                                  is_high=True)

            # low response channel loss
            loss += self._loss_channel(channels=self.channels[i],
                                  grads=hook_module.grads(outputs=-nll_loss),
                                  labels=labels,
                                  is_high=False)
            break
        return loss

    def _loss_channel(self, channels, grads, labels, is_high=True):
        grads =  torch.abs(grads) #F.relu(grads)
        channel_grads = torch.sum(grads, dim=(2, 3)) if not self.flag_3d else torch.sum(grads, dim=(2, 3, 4))  # [batch_size, channels]

        loss = 0
        if is_high:
            for b, l in enumerate(labels):
                loss += (channel_grads[b] * channels[l]).sum() #if l in [2, 3] else torch.tensor(0.).to(channel_grads.device)
        else:
            for b, l in enumerate(labels):
                loss += (channel_grads[b] * (1 - channels[l])).sum() #if l in [2, 3] else torch.tensor(0.).to(channel_grads.device)
        loss = loss / labels.size(0)
        return loss


class GradIntegral:
    def __init__(self, model, modules):
        self.modules = modules
        self.hooks = []

    def add_noise(self):
        for module in self.modules:
            hook = module.register_forward_hook(self._modify_feature_map)
            self.hooks.append(hook)

    def remove_noise(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    # keep forward after modify
    @staticmethod
    def _modify_feature_map(module, inputs, outputs):
        noise = torch.randn(outputs.shape).to(outputs.device)
        # noise = torch.normal(mean=0, std=3, size=outputs.shape).to(outputs.device)
        outputs += noise
        return outputs


class ForwardNurse:
    def __init__(self, module):
        self.handle = module.register_forward_hook(self._modify_output)
        self.channel_weight = None
        
    def _modify_output(self, module, inputs, outputs):
        channel_sum = outputs.sum(dim=(2,3), keepdim=True) #[batch_size, channel, 1, 1]
        channel_mean = channel_sum.mean(dim=1, keepdim=True) #[batch_size, 1, 1, 1]
        self.channel_weight = F.sigmoid(channel_sum-channel_mean) #[batch_size, channel, 1, 1]
        outputs = outputs * self.channel_weight
        return outputs
    
    def remove(self):
        self.handle.remove()


class ForwardDoctor:
    def __init__(self, model, modules, n_classes, alpha=0.999):
        self.model = model
        self.modules = modules
        self.nurses = []
        self.n_classes = n_classes
        self.alpha = alpha
        self.class_weights = [[0]*n_classes for _ in range(len(modules))]

    def assign_nurses(self):
        for module in self.modules:
            self.nurses.append(ForwardNurse(module))

    def remove_nurses(self):
        for nurse in self.nurses:
            nurse.remove()
        self.nurses.clear()

    def loss(self, outputs, labels):
        pred_labels = torch.argmax(outputs, dim=1)
        correct_indices = [torch.arange(labels.size(0))[torch.bitwise_and(pred_labels==labels, labels==i)] for i in range(self.n_classes)]
        loss = 0
        for i, nurse in enumerate(self.nurses):
            channel_weight = nurse.channel_weight
            nurse_loss = 0
            for j in range(self.n_classes):
                if correct_indices[j].size(0) != 0:
                    if isinstance(self.class_weights[i][j], int):
                        self.class_weights[i][j] = torch.mean(channel_weight[correct_indices[j]], dim=0).detach() #[channel, 1, 1]
                    else:
                        self.class_weights[i][j] = self.alpha*self.class_weights[i][j] + (1-self.alpha)*torch.mean(channel_weight[correct_indices[j]], dim=0).detach()
                class_indices = labels == j
                if class_indices.size(0) != 0:
                    class_channel_weight = channel_weight[class_indices] #[class_batch_size, channel, 1, 1]
                    class_channel_weight = (class_channel_weight - self.class_weights[i][j]).squeeze(-1).squeeze(-1)
                    nurse_loss += torch.linalg.vector_norm(class_channel_weight, dim=1, ord=2).sum()
            loss += nurse_loss / labels.size(0)
        return loss / len(self.nurses)

    def visualize_class_weights(self, save_path):
        f, axes = plt.subplots(figsize=(30, 5*len(self.class_weights)), nrows=len(self.class_weights), ncols=1)
        for i, ax in (enumerate(axes) if isinstance(axes, np.ndarray) else enumerate([axes])):
            ax.set_xlabel('channel')
            ax.set_ylabel('category')
            sns.heatmap(torch.stack(self.class_weights[i]).squeeze(-1).squeeze(-1).cpu().numpy(), annot=False, ax=ax)
        plt.savefig(os.path.join(save_path, 'class_weights.png'), bbox_inches='tight')