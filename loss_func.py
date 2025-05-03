import torch
import torch.nn as nn
import torch
import torch.nn as nn
from utils import confidence_weighted_view_fusion

class MLSloss(nn.Module):
    def __init__(self, reduction='mean', scale_factor=1.0):
        super(MLSloss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor  # 缩放因子，用于调整损失范围
        self.MLSloss = nn.MultiLabelSoftMarginLoss(reduction='none')

    def forward(self, main_output, gt_s, flag=None, alpha=0.7, ass_output=None):
        # 损失计算
        main_loss = self.MLSloss(main_output, gt_s)
        if ass_output is None or ass_output == 0 or flag is None:
            return main_loss.mean()

        # 获取beta及b0,b1
        beta = 1 - alpha
        b0 = beta // 2
        b1 = beta // 2

        # 扩散alpha, b0, b1
        alpha_expanded = alpha * torch.ones_like(main_loss)
        b0_expanded = b0 * torch.ones_like(main_loss)
        b1_expanded = b1 * torch.ones_like(main_loss)

        # 根据flag调整加权系数
        main_loss_weighted = main_loss * (alpha_expanded + (b0_expanded * (1-flag[:, 0])) + (b1_expanded * (1-flag[:, 1])))

        # 辅助损失计算
        ol_ass_loss = self.MLSloss(ass_output[0], gt_s)
        sd_ass_loss = self.MLSloss(ass_output[1], gt_s)

        # 对辅助损失进行加权
        ass_loss_weighted = (ol_ass_loss * (b0_expanded * flag[:, 0])) + (sd_ass_loss * (b1_expanded * flag[:, 1]))

        # 合并损失
        total_loss = main_loss_weighted + ass_loss_weighted

        # 计算最终损失
        if self.reduction == 'sum':
            return total_loss.sum()
        elif self.reduction == 'mean':
            return total_loss.mean()
        else:
            return total_loss

class BCELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()

        self.reduction = reduction
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, ol_output, sd_output, gt_s):

        ol_bce_loss = self.loss_fct(ol_output, gt_s)

        sd_bce_loss = self.loss_fct(sd_output, gt_s)

        if self.reduction == 'mean':
            loss = torch.mean(ol_bce_loss + sd_bce_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(ol_bce_loss + sd_bce_loss)
        return loss

class SimpleSumLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(SimpleSumLoss, self).__init__()
        self.reduction = reduction
        self.loss_fct = nn.MultiLabelSoftMarginLoss(reduction='none')
        
    def forward(self, outputs, gt_s, **kwargs):
        # 解包输出 - VanillaModel返回(ol_output, sd_output, fused_output)
        ol_output, sd_output = outputs
        # prediction = max(ol_output, sd_output)

        out = torch.max(ol_output, sd_output)
        # # 计算各个部分的损失
        # ol_loss = self.loss_fct(ol_output, gt_s)
        total_loss = self.loss_fct(out, gt_s)
        
        # # 将所有损失相加
        # total_loss = ol_loss + sd_loss
        
        # 根据reduction模式返回结果
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss