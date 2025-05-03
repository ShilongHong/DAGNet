import torch
from dataset import data_loader, DvX_dataset_collate
from loss_func import BCELoss, MLSloss, SimpleSumLoss
from torch.utils.data import DataLoader, DistributedSampler
from utils import confidence_weighted_view_fusion
from get_ap import AveragePrecisionMeter, AveragePrecisionMeter2
from tqdm import tqdm
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim
import argparse
from opt.ema import ModelEMA
from opt.warmup import LinearWarmup
from thop import profile
from thop import clever_format

from tools.result_to_csv import *
from tools.log_to_tensorboard import *

from model.model_v2 import *
from model.model_DUAL import DUAL
from model.model_AHCR import AHCR
from model.model_VanillaModel import VanillaModel  # 导入新模型

from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
# 模型配置
# 格式: model类型_backbone类型, 例如: model_our_convnext 或 model_AHCR_resnet50
model_name = f'model_ourv3_convnext'
# 从model_name中解析模型类型和backbone类型
model_parts = model_name.split('_')
model_type = '_'.join(model_parts[0:2])  # 获取模型类型 (model_v2 或 model_AHCR)
backbone_type = model_parts[2] if len(model_parts) > 2 else "convnext"  # 默认为convnext

start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
log_head = f"Run time in {start_time}, Model:{model_name}"
update_content = log_head + f"更新内容：实验-{model_name}\n"
base_dir = f'./log/{model_name}_{start_time}/'
print(start_time)
depth_mult = 3
expansion = 2

# 训练参数
start_epoch = 0
epochs = 60
input_shape = [256, 256]
batch_size = 64
num_workers = 32
learning_rate = 1e-4
# learning_rate = 0.005
weight_decay = 1e-3
grad_clip = 5
max_norm = grad_clip
print_freq = 10

# 硬件设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练选项
use_amp = True  # 自动混合精度训练
use_ema = True  # 使用指数移动平均

# 数据路径
train_annotation_path = './data/DvXray_train.txt'
val_annotation_path = './data/DvXray_val.txt'
test_annotation_path = './data/DvXray_test.txt'

# 跟踪最佳结果
best_acc = 0
best_epoch = 0

# 损失函数参数
alpha = 0.7
threshold = 0.5

def setup_distributed(args):
    """初始化分布式训练环境"""
    print(f"设置分布式环境，local_rank: {args.local_rank}")
    
    if args.local_rank == -1:  # 非分布式训练
        print("检测到非分布式模式")
        return False
    
    try:
        # 设置当前GPU设备
        print(f"设置CUDA设备为: {args.local_rank}")
        torch.cuda.set_device(args.local_rank)
        
        # 初始化分布式进程组
        print("使用NCCL后端初始化进程组")
        dist.init_process_group(backend='nccl')
        # 确保每个进程的随机数种子不同，但可预测
        torch.manual_seed(3407 + args.local_rank)
        np.random.seed(3407 + args.local_rank)
        
        # 确认初始化成功
        print(f"进程组初始化成功。Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")
        
        return True
    except Exception as e:
        print(f"初始化分布式环境时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_checkpoint(epoch, model, optimizer, ap, warmup_scheduler, scheduler, is_main_process=True):
    """保存模型检查点 - 只保存模型参数而不是整个模型对象"""
    global best_acc, start_time, best_epoch
    if epoch <= epochs/2:
        return
        
    # 在多GPU训练中同步最佳精度信息
    if dist.is_initialized():
        # 先确保所有进程都走到这一步
        dist.barrier()
        
        # 只有主进程才会有有效的ap值，其他进程接收广播
        ap_tensor = torch.tensor([ap], device=device)
        
        # 从主进程广播ap值到所有进程，而不是使用all_reduce
        dist.broadcast(ap_tensor, 0)
        ap = ap_tensor.item()
        
        # 同步点
        dist.barrier()
    
    if ap > best_acc:
        best_acc = ap
        best_epoch = epoch
        
        # 仅在主进程保存模型
        if is_main_process:
            state = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),  # 只保存模型参数
                'optimizer_state_dict': optimizer.state_dict(),
                'start_time': start_time,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'warmup_scheduler': warmup_scheduler,
                'scheduler': scheduler
            }
            os.makedirs(base_dir, exist_ok=True)
            torch.save(state, base_dir+f'{model_name}_last.pth')   
            torch.save(state, base_dir+f'{model_name}_best.pth')  
    else:
        # 仅在主进程保存模型
        if is_main_process:
            state = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),  # 只保存模型参数
                'optimizer_state_dict': optimizer.state_dict(),
                'start_time': start_time,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'warmup_scheduler': warmup_scheduler,
                'scheduler': scheduler
            }
            os.makedirs(base_dir, exist_ok=True)
            torch.save(state, base_dir+f'{model_name}_last.pth')   
    
    if is_main_process:
        print(f'Best result in epoch {best_epoch+1}, best_accuracy: {best_acc:.4f}')

def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler, warmup_scheduler, is_distributed=False):
    # if dist.is_initialized():
    #     print(f"Process {dist.get_rank()}/{dist.get_world_size()} using GPU: {torch.cuda.current_device()}")
    
    """训练一个epoch"""
    model.train()
    losses = []
    
    # 确定是否为主进程
    is_main_process = not is_distributed or (dist.get_rank() == 0)

    # 使用tqdm显示训练进度条（仅在主进程中）
    if is_main_process:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} Training", ncols=100)
    else:
        pbar = enumerate(train_loader)

    # 遍历批次数据
    for i, (img_ols, img_sds, gt_s, flag) in pbar:
        # 将数据转移到设备
        img_ols = img_ols.to(device)
        img_sds = img_sds.to(device)
        gt_s = gt_s.to(device)
        flag = flag.to(device)

        # 使用混合精度训练
        if scaler is not None:
            with torch.amp.autocast('cuda'):  
                # 区分不同模型的输出处理
                if "AHCR" in model_type:
                    ol_output, sd_output = model(img_ols, img_sds)
                    loss = criterion(ol_output, sd_output, gt_s)
                elif "DUAL" in model_type:
                    output = model(img_ols, img_sds)
                    loss = criterion(output, gt_s)
                elif "Vanilla" in model_type: 
                    outputs = model(img_ols, img_sds)
                    loss = criterion(outputs, gt_s)
                else:
                    outputs = model(img_ols, img_sds)
                    loss = criterion(outputs, gt_s, flag)
            
            # 反向传播与优化
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # 常规训练
            # 区分不同模型的输出处理
            if "AHCR" in model_type:
                ol_output, sd_output = model(img_ols, img_sds)
                loss = criterion(ol_output, sd_output, gt_s)
            elif "DUAL" in model_type:
                output = model(img_ols, img_sds)
                loss = criterion(output, gt_s)
            elif "Vanilla" in model_type:
                outputs = model(img_ols, img_sds)
                loss = criterion(outputs, gt_s)
            else:
                outputs = model(img_ols, img_sds)
                loss = criterion(outputs, gt_s, flag)
                
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        # 记录损失
        cur_loss = loss.item()
        losses.append(cur_loss)
        avg_loss = sum(losses) / len(losses)
        
        # 学习率调整
        cur_lr = learning_rate
        warmup_scheduler.step()  # 更新warmup
        if scheduler is not None:
            cur_lr = scheduler.get_last_lr()[0]
        
        # 更新进度条（仅在主进程）
        if is_main_process and isinstance(pbar, tqdm):
            pbar.set_postfix({'Loss': f'{avg_loss:.3f}', 'lr': f'{cur_lr:.3e}'})
    
    # 更新学习率
    if scheduler is not None:
        scheduler.step()
    
    # 如果是分布式训练，同步损失
    if is_distributed:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    
    return avg_loss

def evaluate(val_loader, model, criterion, log_file=None, is_test=False, is_distributed=False):
    """评估模型 - 简化版本，只在主进程中进行实际评估"""
    model.eval()  # 评估模式，不计算梯度
    
    # 确定是否为主进程
    is_main_process = not is_distributed or (dist.get_rank() == 0)
    
    # 如果使用分布式训练，先确保各进程同步
    if is_distributed:
        # 同步所有进程
        dist.barrier()
    
    # 如果不是主进程，等待主进程完成评估后再获取结果
    if is_distributed and not is_main_process:
        # 创建接收结果的张量
        metrics_tensor = torch.zeros(6, dtype=torch.float32, device=device)
        # 同步点：等待主进程完成评估
        dist.barrier()
        # 接收广播的结果
        dist.broadcast(metrics_tensor, 0)
        # 最终同步点
        dist.barrier()
        # 返回接收到的结果
        return metrics_tensor[0].item(), metrics_tensor[1].item(), metrics_tensor[2].item(), metrics_tensor[3].item(), metrics_tensor[4].item(), metrics_tensor[5].item()
    
    # 下面的代码只在主进程(或非分布式训练)执行
    ap_meter0 = AveragePrecisionMeter()
    ap_meter0.reset()
    
    ap_meter = AveragePrecisionMeter2()
    ap_meter.reset()
    
    losses = []
    
    # 使用独立的评估模型副本，避免DDP内部的同步操作
    eval_model = model.module if isinstance(model, DDP) else model
    
    # 使用tqdm显示验证进度条（仅在主进程中）
    mode_str = "Testing" if is_test else "Validation"
    if is_main_process:
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"{mode_str}", ncols=100)
    else:
        pbar = enumerate(val_loader)
        
    # 评估时禁用梯度计算
    with torch.no_grad():
        for i, (img_ols, img_sds, gt_s, flag) in pbar:
            img_ols = img_ols.to(device)
            img_sds = img_sds.to(device)
            gt_s = gt_s.to(device)

            # 使用非DDP模型进行预测
            # 区分不同模型的输出处理
            if "AHCR" in model_type:
                ol_output, sd_output = eval_model(img_ols, img_sds)
                loss = criterion(ol_output, sd_output, gt_s)
                prediction = confidence_weighted_view_fusion(torch.sigmoid(ol_output), torch.sigmoid(sd_output))
            elif "DUAL" in model_type:
                output = eval_model(img_ols, img_sds)
                loss = criterion(output, gt_s)
                prediction = torch.sigmoid(output)
            elif "Vanilla" in model_type:  # 添加VanillaModel支持
                ol_output, sd_output = eval_model(img_ols, img_sds)
                loss = criterion((ol_output, sd_output), gt_s)
                prediction = confidence_weighted_view_fusion(torch.sigmoid(ol_output), torch.sigmoid(sd_output))
            else:
                main_output = eval_model(img_ols, img_sds)
                loss = criterion(main_output, gt_s)
                prediction = torch.sigmoid(main_output)

            # 计算指标和损失
            ap_meter.add(prediction.data, gt_s)
            if log_file is not None or "only_eval" in globals() and globals()["only_eval"]:
                ap_meter0.add(prediction.data, gt_s)
            
            losses.append(loss.item())
            
            # 更新进度条（仅在主进程）
            if is_main_process and isinstance(pbar, tqdm):
                cur_loss = loss.item()
                pbar.set_postfix({'Loss': f'{cur_loss:.3f}'})

    avg_loss = sum(losses) / len(losses) if losses else 0
    ap, auroc, precision, recall, f1 = ap_meter.value()
    
    # 处理日志输出
    if log_file is not None or "only_eval" in globals() and globals()["only_eval"]:
        if is_test:
            print(f"test_mAP: {ap}, test_precision:{precision}, test_recall:{recall}, test_f1_score:{f1}, test_ROC:{auroc}")
        else:
            print(f"val_mAP: {ap}, val_precision:{precision}, val_recall:{recall}, val_f1_score:{f1}, val_ROC:{auroc}")
        
        each_ap = ap_meter0.value()
        if log_file is not None:
            log_file.write(f"mAP: {ap}, precision:{precision}, recall:{recall}, f1_score:{f1}, ROC:{auroc}\n")
            for item_ap in each_ap:
                log_file.write(f"{item_ap} ")
            log_file.write(f"\n")
        print(each_ap)
        print(f"AP for each class: {each_ap.mean()}")
    
    # 在分布式环境中，主进程广播结果到其他进程
    if is_distributed:
        # 同步点：准备广播结果
        dist.barrier()
        
        # 创建并广播结果张量
        metrics_tensor = torch.tensor([ap, auroc, precision, recall, f1, avg_loss], 
                                     dtype=torch.float32, device=device)
        dist.broadcast(metrics_tensor, 0)
        
        # 最终同步点
        dist.barrier()
    
    return ap, auroc, precision, recall, f1, avg_loss


def main():
    """主函数"""
    args = parse_args()  # 获取命令行参数
    global start_epoch, start_time, num_workers, only_eval, depth_mult, expansion, base_dir, batch_size, learning_rate, model_name, backbone_type, model_type, epochs
    
        
    # 初始化分布式环境
    is_distributed = setup_distributed(args)
    is_main_process = not is_distributed or (dist.get_rank() == 0)
    
    if is_distributed:
        # 如果是分布式训练，调整batch_size和workers
        batch_size = batch_size // dist.get_world_size()
        num_workers = max(num_workers // dist.get_world_size(), 1)
        if is_main_process:
            print(f"Distributed training with {dist.get_world_size()} processes")
            print(f"Adjusted batch size: {batch_size}, workers per process: {num_workers}")
        # learning_rate = dist.get_world_size() * learning_rate # 线性缩放学习率

    checkpoint = args.checkpoint  # 从命令行获取checkpoint
    only_eval = args.eval  # 从命令行获取only_eval

    # 1. 模型初始化阶段
    if checkpoint is None:
        # 根据model_type选择不同的模型
        if "AHCR" in model_type:
            model = AHCR(num_classes=15, backbone=backbone_type)  # 将backbone类型传递给AHCR
        elif "DUAL" in model_type:
            model = DUAL(num_classes=15, dropout=0.3, backbone=backbone_type)
        elif "Vanilla" in model_type:  # 添加VanillaModel支持
            model = VanillaModel(num_classes=15, dropout=0.3, backbone=backbone_type)
        else:
            model = MyModel(num_classes=15, depth_mult=depth_mult, expansion=expansion, backbone=backbone_type)
        
        model = model.to(device)
        
        flops, params = profile(model, (torch.randn(1, 3, 256, 256).to(device), torch.randn(1, 3, 256, 256).to(device)))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Model FLOPs: {flops}, Parameters: {params}")
        # return 0
        
        # 重新初始化模型以进行训练
        if "AHCR" in model_type:
            model = AHCR(num_classes=15, backbone=backbone_type)  # 将backbone类型传递给AHCR
        elif "DUAL" in model_type:
            model = DUAL(num_classes=15, dropout=0.3, backbone=backbone_type)
        elif "Vanilla" in model_type:  # 添加VanillaModel支持
            model = VanillaModel(num_classes=15, dropout=0.3, backbone=backbone_type)
        else:
            model = MyModel(num_classes=15, depth_mult=depth_mult, expansion=expansion, backbone=backbone_type)
            
        model = model.to(device)
        parameters = model.parameters()
        
        optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-8)

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/100)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3080], gamma=0.5)
        warmup_duration = epochs // 10 * 175
        warmup_scheduler = LinearWarmup(scheduler, warmup_duration)
    else:
        if is_main_process:
            print(f"Loading checkpoint: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        
        # 创建一个新的模型实例
        if "AHCR" in model_type:
            model = AHCR(num_classes=15, backbone=backbone_type)  # 将backbone类型传递给AHCR
        elif "DUAL" in model_type:
            model = DUAL(num_classes=15, dropout=0.3, backbone=backbone_type)
        elif "Vanilla" in model_type:  # 添加VanillaModel支持
            model = VanillaModel(num_classes=15, dropout=0.3, backbone=backbone_type)
        else:
            model = MyModel(num_classes=15, depth_mult=depth_mult, expansion=expansion, backbone=backbone_type)
            
        model = model.to(device)
        
        # 只加载模型参数
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        start_epoch = checkpoint_data['epoch'] + 1
        optimizer = optim.AdamW(model.parameters(), 
                            lr=learning_rate,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        start_time = checkpoint_data['start_time']
        best_acc = checkpoint_data['best_acc']
        best_epoch = checkpoint_data['best_epoch']
        warmup_scheduler = checkpoint_data['warmup_scheduler']
        scheduler = checkpoint_data['scheduler']
        base_dir = f'./log/{model_name}_{start_time}/'
    
    # 将模型放到当前设备
    model = model.to(device)
    
    # 封装模型为DDP模型
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                   find_unused_parameters=True)
    
    # 2. 损失函数和优化器设置
    if "AHCR" in model_name:
        criterion = BCELoss().to(device)
    elif "Vanilla" in model_name:  # 添加VanillaModel损失函数支持
        criterion = SimpleSumLoss().to(device)
    else:
        criterion = MLSloss().to(device)
    scaler = None
    ema = None

    # 3. 数据加载
    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    with open(val_annotation_path) as f:
        val_lines = f.readlines()

    with open(test_annotation_path) as f:
        test_lines = f.readlines()
        

    
    # 创建数据加载器，使用DistributedSampler进行分布式训练
    if is_distributed:
        test_sampler = DistributedSampler(data_loader(test_lines, input_shape, is_train=False), shuffle=False)
        test_loader = DataLoader(data_loader(test_lines, input_shape, is_train=False),
                                 batch_size=batch_size//2,
                                 sampler=test_sampler,
                                 drop_last=False,
                                 collate_fn=DvX_dataset_collate,
                                 num_workers=num_workers,
                                 pin_memory=True)

    else:
        # 非分布式训练的数据加载器
        test_loader = DataLoader(data_loader(test_lines, input_shape, is_train=False),
                                 batch_size=batch_size//2,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=DvX_dataset_collate,
                                 num_workers=num_workers,
                                 pin_memory=True)
    
    # 4. 评估模式
    if only_eval:
        if is_distributed and is_main_process:
            # 创建主进程专用的完整测试数据加载器
            main_test_loader = DataLoader(data_loader(test_lines, input_shape, is_train=False),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=DvX_dataset_collate,
                                 num_workers=num_workers,
                                 pin_memory=True)
            
            
            # 使用完整数据集进行评估
            evaluate(main_test_loader, model, criterion, is_test=True, is_distributed=is_distributed)
        else:
            # 非主进程或非分布式训练使用原始数据加载器
            evaluate(test_loader, model, criterion, is_test=True, is_distributed=is_distributed)

        
        # 如果使用分布式训练，在结束时销毁进程组
        if is_distributed:
            dist.destroy_process_group()
        return 0
    
    # 5. 创建训练数据加载器
    if is_distributed:
        train_sampler = DistributedSampler(data_loader(train_lines, input_shape))
        train_loader = DataLoader(data_loader(train_lines, input_shape), 
                                  batch_size=batch_size, 
                                  sampler=train_sampler,
                                  drop_last=False, 
                                  collate_fn=DvX_dataset_collate, 
                                  num_workers=num_workers, 
                                  pin_memory=True)
        
        val_sampler = DistributedSampler(data_loader(val_lines, input_shape, is_train=False), shuffle=False)
        val_loader = DataLoader(data_loader(val_lines, input_shape, is_train=False),
                                batch_size=batch_size, 
                                sampler=val_sampler,
                                drop_last=False, 
                                collate_fn=DvX_dataset_collate, 
                                num_workers=num_workers, 
                                pin_memory=True)
    else:
        train_loader = DataLoader(data_loader(train_lines, input_shape), 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  drop_last=False, 
                                  collate_fn=DvX_dataset_collate, 
                                  num_workers=num_workers, 
                                  pin_memory=True)
        
        val_loader = DataLoader(data_loader(val_lines, input_shape, is_train=False),
                                batch_size=batch_size, 
                                shuffle=False,
                                drop_last=False, 
                                collate_fn=DvX_dataset_collate, 
                                num_workers=num_workers, 
                                pin_memory=True)

    # 6. 其他训练设置
    if use_ema:
        ema = ModelEMA(model)
        
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')

    # 7. 日志和记录设置（仅在主进程）
    writer = None
    csv_filename = None
    log_file = None
    
    if is_main_process:
        os.makedirs(base_dir, exist_ok=True)
        writer = init_tensorboard(log_dir=base_dir)
        csv_filename = init_csv(base_dir + f'results.csv')

        # 打开日志文件
        log_file = open(base_dir + f'log.txt', 'a')
        if checkpoint is None:
            log_file.write(update_content)
            # 写入配置信息
            config_info = f"""Current Configuration:
            Model Name: {model_name}
            Parameters: {params}
            FLOPs: {flops}
            Depth Multiplier: {depth_mult}
            Expansion: {expansion}
            Start Epoch: {start_epoch}
            Total Epochs: {epochs}
            Input Shape: {input_shape}
            Batch Size: {batch_size}
            Workers Per Process: {num_workers}
            Learning Rate: {learning_rate}
            Weight Decay: {weight_decay}
            Gradient Clip: {grad_clip}
            Data Paths:
                Training: {train_annotation_path}
                Validation: {val_annotation_path}
            Evaluation Mode: {only_eval}
            Checkpoint: {checkpoint}
            EMA: {use_ema}
            Mixed Precision: {use_amp}
            Alpha: {alpha}
            Distributed Training: {is_distributed}
            World Size: {dist.get_world_size() if is_distributed else 1}\n"""
        
            log_file.write(config_info)
            log_file.flush()
    
    if is_main_process:
        print("Starting training...")
                # 创建主进程专用的验证数据加载器
    if is_distributed and is_main_process:
        # 创建一个不使用DistributedSampler的数据加载器，确保主进程能看到所有数据
        main_val_loader = DataLoader(data_loader(val_lines, input_shape, is_train=False),
                             batch_size=batch_size, 
                             shuffle=False,
                             drop_last=False, 
                             collate_fn=DvX_dataset_collate, 
                             num_workers=num_workers, 
                             pin_memory=True)
    
    # 8. 训练循环
    for epoch in range(start_epoch, epochs):
        # 设置分布式采样器的epoch
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
            
        # 训练一个epoch
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = train(train_loader, model, criterion, optimizer, epoch, 
                       scaler, scheduler, warmup_scheduler, is_distributed)

        # EMA更新
        if ema is not None:
            ema.update(model)
        
        # 确保所有进程在评估前同步
        if is_distributed:
            dist.barrier()
            
        # 创建主进程专用的验证数据加载器
        if is_distributed and is_main_process:
            val_ap, val_auroc, val_precision, val_recall, val_f1, val_loss = evaluate(
                main_val_loader, model, criterion, is_distributed=is_distributed)
        else:
            # 非分布式训练或非主进程使用原始数据加载器
            val_ap, val_auroc, val_precision, val_recall, val_f1, val_loss = evaluate(
                val_loader, model, criterion, is_distributed=is_distributed)
        
        # 确保所有进程在保存检查点前同步
        if is_distributed:
            dist.barrier()
            
        # 保存检查点（仅在主进程）
        save_checkpoint(epoch, model, optimizer, val_ap, warmup_scheduler, scheduler, is_main_process)
        
        # 打印信息和记录日志（仅在主进程）
        if is_main_process:
            print(f"Epoch {epoch+1}, Train loss: {avg_loss:.4e}, Validation mAP: {val_ap:.4f}, ROC: {val_auroc:.4f}, Loss: {val_loss:.4e}")
            
            # 写入日志
            log_file.write(f"Epoch {epoch+1}, lr:{current_lr:.3e}, train_loss:{avg_loss:.3e}, val_mAP: {val_ap:.4f}, val_precision:{val_precision:.4f}, val_recall:{val_recall:.4f}, val_f1_score:{val_f1:.4f}, val_ROC:{val_auroc:.4f}, val_loss:{val_loss:.4e}\n")
            log_file.flush()  # 强制刷新缓存
            
            # 写入CSV和TensorBoard
            write_to_csv(csv_filename, epoch + 1, avg_loss, 0, current_lr, val_loss, val_ap, val_precision, val_recall, val_f1)
            log_to_tensorboard(writer, epoch + 1, avg_loss, 0, val_loss, val_ap, current_lr, val_precision, val_recall, val_f1)  # 取消注释，启用TensorBoard日志

    # 9. 训练结束后评估
    if is_distributed:
        # 确保所有进程完成训练后再继续
        dist.barrier()
    
    # 最终评估应该仅在主进程中加载最佳模型并进行评估
    if is_main_process:
        print(f"Training completed, started at {start_time}")
        
        # 加载最佳模型进行最终评估
        best_checkpoint_data = torch.load(base_dir+f'{model_name}_best.pth')
        
        # 创建新的模型实例并加载最佳参数
        if "AHCR" in model_type:
            best_model = AHCR(num_classes=15, backbone=backbone_type)  # 将backbone类型传递给AHCR
        elif "DUAL" in model_type:
            best_model = DUAL(num_classes=15, dropout=0.3, backbone=backbone_type)
        elif "Vanilla" in model_type:  # 添加VanillaModel支持
            best_model = VanillaModel(num_classes=15, dropout=0.3, backbone=backbone_type)
        else:
            best_model = MyModel(num_classes=15, depth_mult=depth_mult, expansion=expansion, backbone=backbone_type)
        
        best_model.load_state_dict(best_checkpoint_data['model_state_dict'])
        best_model = best_model.to(device)
        
        # 创建非分布式的数据加载器，确保使用完整的数据集进行评估
        val_full_loader = DataLoader(data_loader(val_lines, input_shape, is_train=False),
                               batch_size=batch_size, 
                               shuffle=False,
                               drop_last=False, 
                               collate_fn=DvX_dataset_collate, 
                               num_workers=num_workers, 
                               pin_memory=True)
        
        test_full_loader = DataLoader(data_loader(test_lines, input_shape, is_train=False),
                                batch_size=batch_size, 
                                shuffle=False,
                                drop_last=False,
                                collate_fn=DvX_dataset_collate,
                                num_workers=num_workers,
                                pin_memory=True)
                                
        
        log_file.write('-' * 10 + 'Validation Results' + '-' * 10 + '\n')
        evaluate(val_full_loader, best_model, criterion, log_file, is_test=False, is_distributed=False)
        
        log_file.write('-' * 10 + 'Test Results' + '-' * 10 + '\n')
        evaluate(test_full_loader, best_model, criterion, log_file, is_test=True, is_distributed=False)
        
        # 关闭日志
        log_file.close()
        if writer:
            writer.close()
    
    # 10. 清理分布式环境
    if is_distributed:
        dist.destroy_process_group()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train and Evaluate Dual-view Model with DDP")
    parser.add_argument("--eval", action="store_true", default=False, 
                        help="Only evaluate the model (default: False)")
    parser.add_argument("-r", "--checkpoint", type=str, default=None, 
                        help="Path to model checkpoint (default: None)")
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help="Local rank for distributed training")
    args = parser.parse_args()

    # 从环境变量中获取LOCAL_RANK (torchrun设置)
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(f"从环境变量获取local_rank: {args.local_rank}")
    return args

if __name__ == '__main__':
    main()