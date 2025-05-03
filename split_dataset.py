import os
import random
import json
import numpy as np

train_test_val_percent = [0.7, 0.1, 0.2]  # 训练集、验证集、测试集比例：7:1:2

base_path = './data'
DvXray_set = [('DvXray', 'train'), ('DvXray', 'test'), ('DvXray', 'val')]  # 添加验证集

prohibited_item_classes = {
    'Gun': 0, 'Knife': 1, 'Wrench': 2, 'Pliers': 3, 'Scissors': 4, 'Lighter': 5, 'Battery': 6,
    'Bat': 7, 'Razor_blade': 8, 'Saw_blade': 9, 'Fireworks': 10, 'Hammer': 11,
    'Screwdriver': 12, 'Dart': 13, 'Pressure_vessel': 14
}

def convert_annotation(image_id, list_file):
    try:
        with open(f'./data/DvXray/{image_id}.json', 'r', encoding='utf-8') as j:
            label = json.load(j)
    except FileNotFoundError:
        print(f"Warning: {image_id}.json not found!")
        return

    gt = np.zeros(15, dtype=int)  # 初始化15个类别
    objs = label['objects']
    ol_bbs = []
    sd_bbs = []

    if objs != 'None':
        for obj in objs:
            ind = prohibited_item_classes[obj['label']]
            gt[ind] = 1  # 对应类别标记为1
            
            # 获取对应的边界框信息
            ol_bb = obj.get('ol_bb', [])
            sd_bb = obj.get('sd_bb', [])

            if ol_bb:
                ol_bbs.append(ol_bb)
            if sd_bb:
                sd_bbs.append(sd_bb)

    # 类别信息，使用 , 作为分隔符，并且加 # 作为前缀
    list_file.write('' + ','.join([str(a) for a in gt]))

    # 只有在存在边界框时才写入，避免多余的 # 和空格
    if ol_bbs:
        list_file.write('#' + ' '.join([str(bb) for bb in ol_bbs]))
    else:
        list_file.write('#[]')  # 如果没有OL边界框，写入空列表，不加空格

    if sd_bbs:
        list_file.write('#' + ' '.join([str(bb) for bb in sd_bbs]))
    else:
        list_file.write('#[]')  # 如果没有SD边界框，写入空列表，不加空格
    # print(ol_bbs)

    if ol_bbs == ["difficult"]:
        return True

    elif sd_bbs == ["difficult"]:
        return True
        
    return False

if __name__ == "__main__":

    # random.seed(3407)
    random.seed(19)

    jsonfilepath = os.path.join(base_path, 'DvXray')
    saveBasePath = os.path.join(base_path, 'split')

    # 检查目录是否存在，若不存在则创建
    if not os.path.exists(saveBasePath):
        os.makedirs(saveBasePath)

    temp_file = os.listdir(jsonfilepath)

    total_json = [js for js in temp_file if js.endswith('.json')]

    num = len(total_json)
    
    # 计算各个数据集的大小
    tr = int(num * train_test_val_percent[0])
    te = int(num * train_test_val_percent[1])
    # te = num - tr - val  # 剩下的作为测试集

    # 随机选择数据集
    random.shuffle(total_json)

    train_images = total_json[:tr]
    test_images = total_json[tr:tr+te]
    val_images = total_json[tr+te:]

    print(f"Train size: {len(train_images)}")
    print(f"Validation size: {len(val_images)}")
    print(f"Test size: {len(test_images)}")

    # 创建训练集、验证集和测试集的文件
    with open(os.path.join(saveBasePath, 'train.txt'), 'w') as ftrain, \
         open(os.path.join(saveBasePath, 'val.txt'), 'w') as fval, \
         open(os.path.join(saveBasePath, 'test.txt'), 'w') as ftest:

        for image in train_images:
            ftrain.write(image[:-5] + '\n')  # 去掉.json后缀
        for image in val_images:
            fval.write(image[:-5] + '\n')
        for image in test_images:
            ftest.write(image[:-5] + '\n')
            
    with open(f'DvXray_diff.txt', 'w', encoding='utf-8') as diff_file:
        # 按照train.txt、val.txt和test.txt的分割写入相应文件
        for name, img_set in DvXray_set:
            # 根据 img_set 分别读取训练集、验证集和测试集
            if img_set == 'train':
                image_ids = open(os.path.join(saveBasePath, 'train.txt'), encoding='utf-8').read().strip().split()
            elif img_set == 'val':
                image_ids = open(os.path.join(saveBasePath, 'val.txt'), encoding='utf-8').read().strip().split()
            elif img_set == 'test':
                image_ids = open(os.path.join(saveBasePath, 'test.txt'), encoding='utf-8').read().strip().split()
    
            # 确保读取到test.txt的数据
            print(f"Processing {img_set} data: {len(image_ids)} images")
            
            with open(f'{name}_{img_set}.txt', 'w', encoding='utf-8') as list_file:    
                for image_id in image_ids:
                    # 写入图像路径
                    list_file.write(f'./data/DvXray/{image_id}_OL.png#')
                    list_file.write(f'./data/DvXray/{image_id}_SD.png#')
    
                    # 转换标注并写入
                    flag = convert_annotation(image_id, list_file)
                    list_file.write('\n')
                    if img_set != "train" and flag is True:
                            print(img_set)
                            # 写入图像路径
                            diff_file.write(f'./data/DvXray/{image_id}_OL.png#')
                            diff_file.write(f'./data/DvXray/{image_id}_SD.png#')
                            convert_annotation(image_id, diff_file)
                            diff_file.write('\n')
