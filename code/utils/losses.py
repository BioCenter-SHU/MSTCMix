import torch
import torch.nn
from torch.nn import functional as F
import numpy as np

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NAMES = [ 'ADI', 'BACK', 'LYM', 'STR', 'DEB', 'MUC', 'TUM','MUS','NORM']


CLASS_NUM = [7338,7381,8144,7315,8037,6163,10033,9489,6100]
alpha=0.25
CLASS_WEIGHT = torch.Tensor([(70000/i)**alpha for i in CLASS_NUM]).cuda()


class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)

class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self,reduction='mean',class_weight=CLASS_WEIGHT):
        self.class_weight = class_weight / class_weight.mean()
        self.base_loss = torch.nn.CrossEntropyLoss(reduction=reduction,weight=self.class_weight) #  
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        # output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output , target.long())



class SemiLoss(object):
    """
    Standard SemiLoss without logit adjustment.
    This version keeps training in the original logit/probability space.
    """

    def __init__(self, lambda_u):
        self.lambda_u = float(lambda_u)

    def linear_rampup(self, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / float(rampup_length), 0.0, 1.0)
        return self.lambda_u * float(current)

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        """
        Args:
            outputs_x: [B, C] logits for labeled samples
            targets_x: [B, C] soft labels (mixup / refinement)
            outputs_u: [B, C] logits for unlabeled samples
            targets_u: [B, C] guessed targets
        """

        # ----- Supervised loss: soft cross-entropy -----
        log_probs_x = F.log_softmax(outputs_x, dim=1)  # [B, C]
        Lx = -torch.mean(torch.sum(targets_x * log_probs_x, dim=1))

        # ----- Unsupervised loss: consistency (MSE) -----
        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.linear_rampup(epoch, warm_up)

class DistLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)   # [B, C]
        probs = log_probs.exp()                    # [B, C]

        if target.dim() == 2:
            target_indices = target.argmax(dim=1)
            pt = torch.sum(probs * target, dim=1)        # [B]
            log_pt = torch.sum(log_probs * target, dim=1)
        else:
            target_indices = target.view(-1)
            pt = probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)
            log_pt = log_probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)
        top2_probs, top2_indices = probs.topk(2, dim=1)
        top1_prob = top2_probs[:, 0]
        top2_prob = top2_probs[:, 1]
        top1_class = top2_indices[:, 0]

        is_correct = top1_class == target_indices

   
        focal_factor = torch.ones_like(pt)

        ratio = (top2_prob / top1_prob).clamp(min=1e-6)
        focal_factor[is_correct] = ratio[is_correct] ** self.gamma


        focal_factor[~is_correct] = (1.0 - pt[~is_correct]).clamp(min=1e-6) ** self.gamma


        loss = - focal_factor * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    


     
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0,reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target, c_weights=None):
        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        if target.dim() == 2:
            pt     = torch.sum(probs * target, dim=1)
            log_pt = torch.sum(log_probs * target, dim=1)
        else:
            target = target.view(-1, 1)
            pt     = probs.gather(1, target).squeeze(1)
            log_pt = log_probs.gather(1, target).squeeze(1)

        final_weights=(1 - pt) ** self.gamma  # [B]

        loss = -final_weights * log_pt  # [B]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [B]


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2 * CLASS_WEIGHT
    return mse_loss



def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
