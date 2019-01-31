import torch.nn as nn
import torch.nn.functional as F

# The building block for ResNet
class BuildingBlock(nn.Module):
    def __init__(self, dim, stride = 1):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        # define shortcut structure
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(dim)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        block_out = F.relu(self.bn1(self.conv1(inputs)))
        block_out = self.bn2(self.conv2(block_out))
        shortcut_out = self.shortcut(inputs)
        out = F.relu(block_out + shortcut_out)
        return out


# The bottleneck for ResNet
class Bottleneck(nn.Module):
    def __init__(self, dim, stride=1):
        super(Bottleneck, self).__init__()
        self.expand_ratio = 4
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=stride,bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim * self.expand_ratio, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(dim * self.expand_ratio)
        self.shortcut = nn.Sequential()
        if stride != 1 or dim != dim * self.expand_ratio:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim * self.expand_ratio, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim * self.expand_ratio)
            )

    def forward(self, inputs):
        bottle_out = F.relu(self.bn1(self.conv1(inputs)))
        bottle_out = F.relu(self.bn2(self.conv2(bottle_out)))
        bottle_out = self.bn3(self.conv3(bottle_out))
        shortcut_out = self.shortcut(inputs)
        out = F.relu(bottle_out + shortcut_out)
        return out



# ResNet
class ResNet(nn.Module):

    def __init__(self, in_dim = 3, out_dim = 10, block_config = (3, 4, 23, 3), block = Bottleneck):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = self.build_block(block_config[0], block, 64, 1)
        self.conv3 = self.build_block(block_config[1], block, 128, 2)
        self.conv4 = self.build_block(block_config[2], block, 256, 2)
        self.conv5 = self.build_block(block_config[3], block, 512, 2)
        self.ap = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512, out_dim)

    def build_block(self, block_num, block, dim, stride):
        layers = []
        for i in range(block_num):
            layers.append(block(dim, stride))
        block_unit = nn.Sequential(*layers)
        return block_unit

    def forward(self, inputs):
        conv1_out = self.mp(self.bn1(self.conv1(inputs)))
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv_out = self.ap(conv5_out)
        conv_out = conv_out.view(conv_out.size(0), -1)
        cls_out = self.fc(conv_out)
        return cls_out


def res_net_18(in_dim = 3, out_dim = 10):
    return ResNet(in_dim, out_dim, (2, 2, 2, 2), BuildingBlock)

def res_net_34(in_dim = 3, out_dim = 10):
    return ResNet(in_dim, out_dim, (3, 4, 6, 3), BuildingBlock)

def res_net_50(in_dim = 3, out_dim = 10):
    return ResNet(in_dim, out_dim, (3, 4, 6, 3), Bottleneck)

def res_net_101(in_dim = 3, out_dim = 10):
    return ResNet(in_dim, out_dim, (3, 4, 23, 3), Bottleneck)

def res_net_152(in_dim = 3, out_dim = 10):
    return ResNet(in_dim, out_dim, (3, 8, 36, 3), Bottleneck)




