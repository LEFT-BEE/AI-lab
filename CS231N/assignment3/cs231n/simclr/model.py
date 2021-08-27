import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.f = []
        for name, module in resnet50().named_children():
          #ResNet에서 layer들을 가져와 f에 추가해준다 f는 x의 특징을 추출하는 base encoder
          #이떄 ResNet의 크기를 4배더키운 모델이 더좋은 성능을 보인다고한다 
          #base encoder가 deep하고 widely할수록 성능이 올라간다.
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
              #istnastance자료형이 맞는지 확인해주는 함수 linear와 maxpool2d는 제외해주었다...? 왜지
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
                               #비선형 투영이 매우 매우 중요하다고 한다.

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
