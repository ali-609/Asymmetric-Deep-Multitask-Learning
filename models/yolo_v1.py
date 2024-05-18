import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import *
# from util_layers import Flatten
from torchvision.models import resnet50,resnet101


class YOLOv1(nn.Module):
    def __init__(self, num_bboxes=10, num_classes=14, bn=True):
        super(YOLOv1, self).__init__()
        


        self.feature_size = 8
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        backbone_model = resnet50()

        # self.features = nn.Sequential(
        #     backbone_model.conv1,
        #     backbone_model.bn1,
        #     backbone_model.relu,
        #     backbone_model.maxpool,
        #     backbone_model.layer1,
        #     backbone_model.layer2,
        #     backbone_model.layer3,
        #     backbone_model.layer4
        # )
        self.features=DarkNet(conv_only=True, bn=False, init_weight=True)
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes
        self.conv_layers = nn.Sequential(
                # nn.Conv2d(1024, 512, 3, padding=1),
                # nn.LeakyReLU(0.1),
                nn.Conv2d(1024, 512, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(512, 512, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(512, 190, 3, padding=1),
                # nn.Sigmoid()
            )
        self.fc_layers =   nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            # nn.Dropout(0.5, inplace=False), # is it okay to use Dropout with BatchNorm?
            nn.Linear(512, S * S * (B * (C + 5))),
            nn.Sigmoid()
        )

        self.fc_layers_conv_alter =   nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            # nn.Dropout2d(0.5), # You can use Dropout with convolutional layers
            nn.Conv2d(in_channels=512, out_channels=S * S * (B * (C + 5)), kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        x = self.features(x)

        x = self.conv_layers(x)
        # print(x.shape)
        # exit()
        # x = self.fc_layers(x)

        # x = x.view(-1, B * (5 + C),S, S )
        return x



def test():
    from torch.autograd import Variable

    # Build model with randomly initialized weights
    # darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
    yolo = YOLOv1()

    # Prepare a dummy image to input
    image = torch.rand(1, 3, 1024, 1024)
    image = Variable(image)

    # Forward
    output = yolo(image)
    # Check ouput tensor size, which should be [10, 7, 7, 30]
    print(output.size())


if __name__ == '__main__':
    test()