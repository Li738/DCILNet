import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


"""
loss functions--hybrid loss conducted on all models to make the experiment fair.
"""




class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # print(input.shape)
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))



        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.to(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_loss(logits, true, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        a=torch.eye(num_classes).cuda()
        true_1_hot = a[true.type(torch.long).squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)




class hybrid_loss(nn.Module):
    def __init__(self, gamma=0.5):
        super(hybrid_loss, self).__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=None)
    def forward(self,predictions, target):
        bce = self.focal(predictions, target.squeeze(1))
        dice = dice_loss(predictions, target.squeeze(1))
        loss = dice + bce
        return loss

# def hybrid_loss(predictions, target):
#     loss = 0
#     # focal = FocalLoss(gamma=0, alpha=None)
#     focal = FocalLoss(gamma=0, alpha=None)
#     for prediction in predictions:
#         bce = focal(prediction, target)
#         dice = dice_loss(prediction, target)
#         loss = dice + bce
#     return loss

def hybrid_lossHCX(predictions, target):
    loss = 0
    focal_criterion = FocalLoss(gamma=0.5, alpha=None)
    bce_criterion= FocalLoss(gamma=0, alpha=None)
    for prediction in predictions:
        focal =  focal_criterion(prediction, target)
        # dice = dice_criterion(prediction, target)
        dice = dice_loss(prediction, target)
        bce = bce_criterion(prediction, target)
        focal_new=focal *((bce/focal).detach())
        total_loss = dice + focal_new
    return total_loss
