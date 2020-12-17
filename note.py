#12------------------------------------------------

class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.layer1 = LayerActivation(self.backbone.layer1, 2)
        self.layer2 = LayerActivation(self.backbone.layer2, 3)
        self.layer3 = LayerActivation(self.backbone.layer3, 5)

        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        self.classify = nn.Sequential(
            Linear(1000, 512),
            nn.Linear(512, 24),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classify(x)

        layer1 = self.flatten(self.layer1.features)
        layer2 = self.flatten(self.layer2.features)
        layer3 = self.flatten(self.layer3.features)
        
        return layer1, layer2, layer3, x


class Auxiliary(nn.Module):
    def __init__(self):
        super(Auxiliary, self).__init__()
        self.classify1 = nn.Sequential(
            nn.Linear(64, 24),
            nn.Sigmoid()
        )
        self.classify2 = nn.Sequential(
            nn.Linear(128, 24),
            nn.Sigmoid()
        )
        self.classify3 = nn.Sequential(
            Linear(256, 128),
            nn.Linear(128, 24),
            nn.Sigmoid()
        )

    def forward(self, layer1, layer2, layer3):
        layer1 = self.classify1(layer1)
        layer2 = self.classify2(layer2)
        layer3 = self.classify3(layer3)

        return layer1, layer2, layer3


#13------------------------------------------------
class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.layer1 = LayerActivation(self.backbone.layer1, 2)
        self.layer2 = LayerActivation(self.backbone.layer2, 3)
        self.layer3 = LayerActivation(self.backbone.layer3, 5)
        self.layer4 = LayerActivation(self.backbone.layer4, 2)

        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        
        self.conv12 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.conv12d = Conv2d(256, 128, kernel_size=1)

        self.conv23 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.conv23d = Conv2d(512, 256, kernel_size=1)

        self.conv34 = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.conv34d = Conv2d(1024, 512, kernel_size=1)

        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(512, 512),
            nn.Linear(512, 24),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.backbone(x)
        layer1 = self.layer1.features
        layer2 = self.conv12d(torch.cat([self.conv12(layer1), self.layer2.features], dim=1))
        layer3 = self.conv23d(torch.cat([self.conv23(layer2), self.layer3.features], dim=1))
        layer4 = self.conv34d(torch.cat([self.conv34(layer3), self.layer4.features], dim=1))

        x = self.classify(layer4)
       
        return layer1, layer2, layer3, x


class Auxiliary(nn.Module):
    def __init__(self):
        super(Auxiliary, self).__init__()
        self.classify1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(64, 24),
            nn.Sigmoid()
        )
        self.classify2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(128, 24),
            nn.Sigmoid()
        )
        self.classify3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(256, 128),
            nn.Linear(128, 24),
            nn.Sigmoid()
        )

    def forward(self, layer1, layer2, layer3):
        layer1 = self.classify1(layer1)
        layer2 = self.classify2(layer2)
        layer3 = self.classify3(layer3)

        return layer1, layer2, layer3