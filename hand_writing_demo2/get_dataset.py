import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

repeat_num = 1
# file_list = [
#     'train_sets/3.json',
#     'train_sets/4.json',
#     'train_sets/5.json'
# ]

file_list = [
    # 'test_sets/3.json',
    'test_sets/6.json'
]


data3 = []
label3 = []
img_list = []


def get_set(name):
    # train_set_name = 'test_sets/4.json'
    train_set_name = name
    pt1 = [0., 0.]
    pt2 = [1960. - 194., 491. - 315.]
    col_num = 10
    row_num = 10
    pts0 = []
    pts = []
    edge = (pt2[0] - pt1[0]) / 10
    dy = (2675. - 315.) / 9
    for i in range(col_num):
        pts0.append([[pt1[0] + i * edge, pt1[1]], [pt1[0] + (i + 1) * edge, pt2[1]]])

    for i in range(row_num):
        temp = []
        for item in pts0:
            temp.append([[item[0][0], item[0][1] + i * dy], [item[1][0], item[1][1] + i * dy]])
        pts.append(temp)

    # for r in pts:
    #     print(r)

    with open(train_set_name, 'r') as f:
        data = json.load(f)
        data = data['json']

    data1 = []
    for i in range(row_num):
        temp = []
        for j in range(col_num):
            temp.append([])
        data1.append(temp)

    k1 = (1960. - 194.) / (163.96 - 15.6)
    k2 = (2852. - 315.) / (240.73 - 26.3)

    # sample_num = 8

    for row in data:
        row1 = []
        center_x = 0.
        center_y = 0.
        cnt = 0
        for item in row:
            item['x'] -= 15.6
            item['y'] -= 26.3
            item['x'] *= k1
            item['y'] *= k2
            row1.append([item['x'], item['y']])
            center_x += item['x']
            center_y += item['y']
            cnt += 1
        center_x = center_x / cnt
        center_y = center_y / cnt
        # print(center_x, center_y)
        # row2 = row1[::int(len(row1)/(sample_num - 1))]
        row2 = row1
        # print(row2)
        for i in range(row_num):
            for j in range(col_num):
                pt = pts[i][j]
                if pt[0][0] < center_x < pt[1][0] and pt[0][1] < center_y < pt[1][1]:
                    data1[i][j].append(row2)

    for i in range(row_num):
        for j in range(col_num):
            pt = pts[i][j]
            for scale in data1[i][j]:
                for item in scale:
                    item[0] -= pt[0][0]
                    item[1] -= pt[0][1]

    for i in range(row_num):
        for j in range(col_num):
            data2 = []
            for row in data1[i][j]:
                for item in row:
                    data2.append(item)
            if len(data2) <= 0:
                continue
            while len(data2) < 200:
                data2.append([0., 0.])
            data2_ts = torch.tensor(data2, dtype=torch.float32).t()
            data3.append(data2_ts.tolist())
            img_list.append(data2_ts.tolist())
            # label3.append(1)
            if i < row_num / 2:
                label3.append(1)
            else:
                label3.append(0)


for name in file_list:
    get_set(name)

# k = 64./(edge * 1.2)
idx = 0
for ch in img_list:
    plt.clf()
    plt.scatter(ch[0], ch[1])
    plt.savefig(f'pics/{idx}.png')
    idx += 1

data_tensor = torch.tensor(data3, dtype=torch.float32).repeat(repeat_num, 1, 1)
label_tensor = torch.tensor(label3, dtype=torch.long).repeat(repeat_num)

scale_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=(64, 64),
        scale=(0.9, 1.1),
        ratio=(1.0, 1.0),  # 保持图像的宽高比不变
        interpolation=transforms.InterpolationMode.NEAREST  # 使用最近邻插值
    ),
    transforms.ToTensor()  # 将PIL图像转换回Tensor
])
# data_tensor = torch.stack([scale_transform(TF.to_pil_image(img)) for img in data_tensor])

indices = torch.randperm(data_tensor.size(0))
# data_tensor = data_tensor[indices]
# label_tensor = label_tensor[indices]
print(len(data_tensor))
print(data_tensor.shape)
print(label_tensor)
dataset = TensorDataset(data_tensor, label_tensor)
train_loader = DataLoader(dataset, batch_size=20, shuffle=True, drop_last=True)
test_train_loader = DataLoader(dataset, batch_size=20, shuffle=True, drop_last=True)
