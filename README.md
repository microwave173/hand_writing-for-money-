# 食用方法
### 数据集的制作
田字格纸的上5行会被标为正确，下5行会标为错误

### 上传笔迹
在/server/server.py

启动之后，把安卓端的url改成自己电脑的局域网ip...即可，例如192.168.1.1:5000/postjson

上传之后json会存在json_data中

之后玩家需要自行将json_data中的数据复制到test_sets和train_sets

### 训练
train.py直接运行就可以开始训练

get_dataset.py用于将json包装成DataLoader

get_dataset.py中的file_list和repeat_num需要自行修改，这里面的所有json将被混为一个大的数据集，repeat_num为数据集重复的次数

### 测试
修改get_dataset.py的file_list和repeat_num后直接运行test1.py即可