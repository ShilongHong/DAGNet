import torch
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AveragePrecisionMeter(object):

    def __init__(self):
        super(AveragePrecisionMeter, self).__init__()

        self.reset()

    def reset(self):
        self.scores = torch.empty(0, dtype=torch.float32).to(device)  # 确保是 Tensor
        self.targets = torch.empty(0, dtype=torch.int64).to(device)  # 确保是 Tensor


    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
    
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column per class)'
    
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'
    
        # 使用 torch.cat 拼接张量
        self.scores = torch.cat((self.scores, output), dim=0)
        self.targets = torch.cat((self.targets, target), dim=0)

    def value(self):
        # if self.scores.nume() == 0:
        #     return 0

        ap = torch.zeros(self.scores.size(1))

        for k in range(self.scores.size(1)):

            scores = self.scores[:, k]
            targets = self.targets[:, k]

            ap[k] = AveragePrecisionMeter.average_precision(scores, targets)
        return ap

    @staticmethod
    def average_precision(output, target):
        sorted, indices = torch.sort(output, dim=0, descending=True)

        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if label == 0:
                total_count += 1
            if label == 1:
                pos_count += 1
                total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= (pos_count + 1e-10)
        return precision_at_i

import torchmetrics

class AveragePrecisionMeter2(object):
    def __init__(self):
        super(AveragePrecisionMeter2, self).__init__()
        self.reset()

    def reset(self):
        # 针对多标签任务初始化指标计算器，假设有15个类别
        device = "cuda:0"
        self.ap_calculator = torchmetrics.AveragePrecision(num_labels=15, average='macro', task='Multilabel', sync_on_compute=False).to(device)
        self.precision_calculator = torchmetrics.Precision(num_labels=15, average='macro', task='Multilabel', sync_on_compute=False).to(device)
        self.recall_calculator = torchmetrics.Recall(num_labels=15, average='macro', task='Multilabel', sync_on_compute=False).to(device)
        self.f1_calculator = torchmetrics.F1Score(num_labels=15, average='macro', task='Multilabel', sync_on_compute=False).to(device)
        self.AUROC = torchmetrics.AUROC(num_labels=15, average='macro', task='Multilabel', sync_on_compute=False).to(device)
    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
        if target.dtype != torch.long:
            target = target.long()

        # 更新所有指标
        self.ap_calculator.update(output, target)
        self.precision_calculator.update(output, target)
        self.recall_calculator.update(output, target)
        self.f1_calculator.update(output, target)
        self.AUROC.update(output, target)  # 更新AUROC计算器

    def value(self):
        # 获取并返回所有指标的计算结果
        average_precisions = self.ap_calculator.compute()
        # for idx, ap in enumerate(average_precisions):
        #     print(f"Class {idx}: AP = {ap.item():.4f}")

        precision = self.precision_calculator.compute()
        recall = self.recall_calculator.compute()
        f1 = self.f1_calculator.compute()
        auroc = self.AUROC.compute()
        # print(f"auroc:{auroc}")
        # print(f"Average Precision (AP): {ap}")
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")
        # print(f"F1 Score: {f1}")
        
        return average_precisions, auroc, precision, recall, f1
