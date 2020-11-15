import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module): # 创建一个神经网络的类

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        # 输入为四通道图片 RGBA？
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False) # 通道数 输出深度 卷积核大小 步长
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512) # 一个全连接层 深度64 大小7*7 输出维度为512
        self.__fc2 = nn.Linear(512, action_dim) # 输出维度为行动的个数
        self.__device = device # 使用的设备 cuda或cpu？

    def forward(self, x):
        x = x / 255. # 对像素点进行处理，归一化
        x = F.relu(self.__conv1(x)) # 卷积+激活函数
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1))) # 全连接层+激活
        return self.__fc2(x)

    # 类的静态方法，可以不用创建类实例就调用
    @staticmethod
    def init_weights(module): # 对权重进行初始化
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
