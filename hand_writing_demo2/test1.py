import torch
import torch.nn as nn
import torch.optim as optim
import get_dataset
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=2):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        # 初始化为单位矩阵
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 添加一个小的偏移量，以确保不会有奇异矩阵
        iden = torch.eye(self.k).view(1, self.k * self.k).repeat(batchsize, 1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=2)
        self.feature_transform = TNet(k=64)

        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        num_points = x.size()[2]

        trans = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        point_features = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        # 通过log_softmax输出多分类的概率分布
        x = self.log_softmax(x)
        return x, trans, trans_feat


test_x = get_dataset.data_tensor
test_y = get_dataset.label_tensor


def test_all(ch):
    model = torch.load(f'{ch}.pth')
    with torch.no_grad():
        out, _, _ = model(test_x)
        _, pred = torch.max(out.data, 1)
        score = (pred == test_y).sum().item()
        print(test_y)
        print(pred)
        for i in range(len(test_x)):
            if int(pred[i]) != int(test_y[i]):
                print(i, ",ans: ", test_y[i], ",pred: ", pred[i])
        print("正确率: ", score / len(test_y))


def test_ch(ch, idx):
    model = torch.load(f'{ch}.pth')
    test_data = torch.tensor([test_x[idx].tolist()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out, _, _ = model(test_data)
        _, pred = torch.max(out.data, 1)
        return pred[0] == 1


ans_list = ['半', '半', '半', '奥', '奥', '奥', '己', '己', '己', '候', '候', '候', '大', '大', '大']
res_list = []
for i in range(len(ans_list)):
    res = test_ch(ans_list[i], i).item()
    res_list.append(res)

for i in range(len(res_list)):
    out_ch = '对'
    if not res_list[i]:
        out_ch = '错'
    print(out_ch, end=' ')
    if (i + 1) % 3 == 0:
        print('')
