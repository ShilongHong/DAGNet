from PIL import Image
import numpy as np
import cv2
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import cvtColor, preprocess_input
from timm.data import create_transform
from transformers import BertTokenizer, CLIPProcessor, AutoImageProcessor
# import transformers
class data_loader(Dataset):

    def __init__(self, annotation_lines, input_shape, is_train = True, is_transfom = True):
        super(data_loader, self).__init__()

        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)

        self.input_shape = input_shape
        self.random_erasing = transforms.RandomErasing(p=0.25, value=0)
        self.flip = transforms.RandomHorizontalFlip(p=0.25)
        self.is_train = is_train

        if is_transfom:
            self.transform = build_transform(is_train, input_shape[0])
            # self.transform = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        else:
            self.transform = None
            
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        line = self.annotation_lines[index].split('#')  # 使用 # 分隔符

        # 获取图像路径
        image_ol_path = line[0].strip()  # OL 图像路径
        image_sd_path = line[1].strip()  # SD 图像路径

        # 加载图像
        image_ol = Image.open(image_ol_path).convert('RGB')
        image_sd = Image.open(image_sd_path).convert('RGB')

        # out_path = image_ol_path.split('/')[-1].split('.')[0].split('_')[0]
        # # 保存为新的文件
        # image_ol.save("./img/"+out_path+"image_ol_ori.png")
        # image_sd.save("./img/"+out_path+"image_sd_ori.png")
        # 应用图像增强
        if self.transform:
            # seed = random.randint(0, 100000)
            # torch.manual_seed(seed)
            # random.seed(seed)
            image_ol = self.transform(image_ol)
            image_sd = self.transform(image_sd)
            # from torchvision.utils import save_image
            # # 假设 image_ol 和 image_sd 的形状为 [3, 256, 256]，已经是标准的 (C, H, W) 格式
            # save_image(image_ol, "./img/"+out_path+"image_ol.png")
            # save_image(image_sd, "./img/"+out_path+"image_sd.png")
            # import time
            # time.sleep(100)
            # aaaaa

        # 获取标签（类别 one-hot 编码）
        gt = np.array(list(map(int, line[2].split(','))))
        flag = [1, 1]
        # print(line[3])
        # # 获取边界框数据（需要处理字符串并解析为列表）
        ol_bboxes = line[3].strip() if len(line) > 3 else "[]"
        if "difficult" in ol_bboxes:
            flag[0] = 0
        # print(process_coordinates(ol_bboxes))
        sd_bboxes = line[4].strip() if len(line) > 4 else "[]"
        if "difficult" in sd_bboxes:
            flag[1] = 0
        # # 将边界框字符串转换为列表
        # ol_bboxes = np.array(eval(ol_bboxes)) if ol_bboxes != "[]" else np.zeros(4, dtype=int)
        # sd_bboxes = np.array(eval(sd_bboxes)) if sd_bboxes != "[]" else np.zeros(4, dtype=int)



        return image_ol, image_sd, gt, flag

    def get_random_data(self, image, hue=.1, sat=0.7, val=0.4):

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data
    
def DvX_dataset_collate(batch):
    img_ols, img_sds, gt_s, input_ids, attention_masks, flags = [], [], [], [], [], []
    for img_ol, img_sd, gt, flag in batch:
        img_ols.append(img_ol)
        img_sds.append(img_sd)
        gt_s.append(gt)
        flags.append(flag)
        
    # 转换为张量
    img_ols = torch.stack(img_ols)
    img_sds = torch.stack(img_sds)
    gt_s_np = np.array(gt_s)
    gt_s = torch.tensor(gt_s_np, dtype=torch.float32)
    flag_s_np = np.array(flags)
    flag_s = torch.tensor(flag_s_np, dtype=torch.int32)
    return img_ols, img_sds, gt_s, flag_s

def build_transform(is_train, TARGET_SIZE=224):
    # 固定插值方式
    INTERPOLATION = transforms.InterpolationMode.BICUBIC
    # 训练增强参数
    COLOR_JITTER = 0.4
    AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
    REPROB = 0.25
    # 数据集的均值和标准差
    MEAN = [0.91584104, 0.9297611, 0.939562]
    STD = [0.22090791, 0.1861283, 0.1651021]

    if is_train:
        # 训练阶段的变换
        input_size = 256  # config.DATA.IMG_SIZE
        color_jitter = 0.4  # config.AUG.COLOR_JITTER
        auto_augment = 'rand-m9-mstd0.5-inc1'  # config.AUG.AUTO_AUGMENT
        re_prob = 0.25  # config.AUG.REPROB
        re_mode = 'pixel'  # config.AUG.REMODE
        re_count = 1 # config.AUG.RECOUNT
        interpolation = 'bicubic'  # config.DATA.INTERPOLATION

        transform = create_transform(
            input_size=input_size,
            is_training=is_train,
            color_jitter=color_jitter if color_jitter > 0 else None,
            auto_augment=auto_augment if auto_augment != 'none' else None,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            interpolation=interpolation,
        )
        transform.transforms.append(transforms.Normalize(mean=MEAN, std=STD))

        return transform

    # 测试阶段的变换
    transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=INTERPOLATION),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    return transform
    
def apply_sync_transform(image1, image2, transform):
    # 生成一个随机种子
    seed = random.randint(0, 100000)
    
    # 应用相同的随机变换到两个视角
    random.seed(seed)
    torch.manual_seed(seed)
    transformed_image1 = transform(image1)
    
    random.seed(seed)
    torch.manual_seed(seed)
    transformed_image2 = transform(image2)
    
    return transformed_image1, transformed_image2
    
def process_coordinates(coordinates):
    # 去除空格并检查是否为空或仅包含 []
    coordinates = coordinates.strip()
    
    if not coordinates or coordinates == "[]":
        # 如果为空或者是只有 []
        return ()
    
    # 判断输入中是否包含两个矩形框（根据空格分割）
    if " " in coordinates:
        # 分割字符串，按空格分隔
        boxes = coordinates.split(" ")
        
        # 处理每个框，去除方括号并分割坐标
        box1 = list(map(int, boxes[0][1:-1].split(", ")))  # 去掉方括号并分割
        box2 = list(map(int, boxes[1][1:-1].split(", ")))
        
        return box1, box2
    else:
        # 只有一个框
        box1 = list(map(int, coordinates[1:-1].split(", ")))  # 去掉方括号并分割
        return box1,