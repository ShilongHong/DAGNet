from torch.utils.tensorboard import SummaryWriter
import subprocess
from time import sleep

# 初始化 TensorBoard writer
def init_tensorboard(log_dir='./logs'):
    """初始化 TensorBoard"""
    writer = SummaryWriter(log_dir)
    subprocess.Popen(['tensorboard', '--logdir=./log', '--port=6006'])
    sleep(0.5)
    return writer

# 记录训练过程中的数据
def log_to_tensorboard(writer, epoch, avg_loss, train_ap, val_loss, val_ap, train_lr, val_precision, val_recall, val_f1):
    """将训练过程中的数据记录到 TensorBoard"""
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('mAP/train', train_ap, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('mAP/val', val_ap, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)
    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('Learning Rate', train_lr, epoch)