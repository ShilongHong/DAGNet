import pandas as pd
from openpyxl import Workbook, load_workbook
import torch
import os

def init_csv(filename):
    """ 初始化 CSV 文件，如果文件不存在则创建，并写入表头 """
    if not os.path.exists(filename):
        # 如果文件不存在，创建新的 CSV 文件并添加表头
        df = pd.DataFrame(columns=[
            'Epoch', 'Train_Loss', 'Train_mAP', 'Train_LR', 
            'Val_Loss', 'Val_mAP', 'Val_Precision', 'Val_Recall', 'Val_F1'
        ])
        df.to_csv(filename, index=False)
    return filename
    
def write_to_csv(filename, epoch, avg_loss, train_ap, current_lr, val_loss, val_ap, val_precision, val_recall, val_f1):
    """
    将训练结果写入 CSV 文件

    参数：
    - filename: CSV 文件路径
    - epoch: 当前训练周期
    - avg_loss, train_ap, current_lr, val_loss, val_ap, val_precision, val_recall, val_f1: 训练和验证的相关指标
    """
    
    def to_python_value(value):
        """ 将 tensor 转换为 Python 标量（float），如果已经是 float，则直接返回 """
        if isinstance(value, torch.Tensor):
            return value.item()  # 如果是 tensor，使用 .item() 转换为 Python 标量
        return value  # 如果已经是 float 或其他类型，直接返回

    # 将数据转换为适合写入 CSV 的格式
    epoch = to_python_value(epoch)
    avg_loss = to_python_value(avg_loss)
    train_ap = to_python_value(train_ap)
    current_lr = to_python_value(current_lr)
    val_loss = to_python_value(val_loss)
    val_ap = to_python_value(val_ap)
    val_precision = to_python_value(val_precision)
    val_recall = to_python_value(val_recall)
    val_f1 = to_python_value(val_f1)

    # 将数据转换为 DataFrame 并追加到 CSV 文件
    data = pd.DataFrame([[
        epoch, avg_loss, train_ap, current_lr, val_loss, val_ap, val_precision, val_recall, val_f1
    ]], columns=[
        'Epoch', 'Train_Loss', 'Train_mAP', 'Train_LR', 
        'Val_Loss', 'Val_mAP', 'Val_Precision', 'Val_Recall', 'Val_F1'
    ])
    
    # 追加数据到 CSV 文件
    data.to_csv(filename, mode='a', header=False, index=False)
