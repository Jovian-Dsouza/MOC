import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import math 

def _fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def _fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def _make_upsampling_layer(in_channels, 
                           num_layers=3, 
                           channels=[256, 128, 64],
                           kernel_size=4,
                           padding = 1,
                           output_padding = 0
                           ):
        layers = []
        for i in range(num_layers):
            planes = channels[i]
            conv = nn.Conv2d(in_channels, planes,
                                 kernel_size=(3,3), stride=1, padding=1)

            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            _fill_up_weights(up)
            
            layers.append(
                nn.Sequential(
                    conv,
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    up,
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = planes

        return nn.Sequential(*layers)

def _make_up_layer(in_channels,
                  out_channels,
                  kernel_size=4,
                  padding = 1,
                  output_padding = 0):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)

    up = nn.ConvTranspose2d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=2,
        padding=padding,
        output_padding=output_padding,
        bias=False)
    _fill_up_weights(up)
    
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        up,
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
        
class ResBackbone(nn.Module):
    '''
    Resnet 18 backbone
    '''
    def __init__(self):
        super().__init__()
    
        self.resnet = torchvision.models.resnet18(pretrained=True, progress=False)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.up = _make_upsampling_layer(in_channels=512)
    
    def forward(self, x):
        x = self.resnet(x) # (-1, 512, 6, 8) 44.4MB
        x = self.up(x) # (-1, 64, 48, 64) 58.41MB
        return x

class MNASBackbone(nn.Module):
    '''
    MNASNet 1 Backbone
    '''
    def __init__(self):
        super().__init__()
    
        self.cnn = torchvision.models.mnasnet1_0(pretrained = True, progress=False).layers
        self.up = _make_upsampling_layer(in_channels=1280)
    
    def forward(self, x):
        x = self.cnn(x) # (-1, 1280, 6, 8) 12.41MB
        x = self.up(x) # (-1, 64, 48, 64) 31.19MB
        return x

class ResSkipBackbone(nn.Module):
    '''
    Resnet 18 backbone with skip connections
    '''
    def __init__(self):
        super().__init__()
    
        self.resnet = torchvision.models.resnet18(pretrained=True, progress=False)
        self.pre = nn.Sequential(*(list(self.resnet.children())[:4]))
        self.layer1 = nn.Sequential(*(list(self.resnet.children())[4]))
        self.layer2 = nn.Sequential(*(list(self.resnet.children())[5]))
        self.layer3 = nn.Sequential(*(list(self.resnet.children())[6]))
        self.layer4 = nn.Sequential(*(list(self.resnet.children())[7]))

        self.up1 = _make_up_layer(in_channels=512, out_channels=256)
        self.up2 = _make_up_layer(in_channels=512, out_channels=128)
        self.up3 = _make_up_layer(in_channels=256, out_channels=64)
    
    def forward(self, x):
        x = self.pre(x) # (-1, 64, 48, 64)
        d1 = self.layer1(x)  # (-1, 64, 48, 64)
        d2 = self.layer2(d1)  # (-1, 128, 24, 32)
        d3 = self.layer3(d2)  # (-1, 256, 12, 16)
        d4 = self.layer4(d3)  # (-1, 512, 6, 8)

        u1 = self.up1(d4) # (-1, 256, 12, 16)
        u2 = self.up2(torch.cat([u1, d3], 1)) # (-1, 128, 24, 32)
        u3 = self.up3(torch.cat([u2, d2], 1)) # (-1, 64, 48, 64)

        return u3

class MOC_Net(pl.LightningModule):
    def __init__(self, arch, num_classes, K, head_conv=256):
        super().__init__()
        self.K = K
        if arch == 'resnet':
            self.backbone = ResBackbone()
        elif arch == 'mnasnet':
            self.backbone = MNASBackbone()
        elif arch == 'resnet_skip':
            self.backbone = ResSkipBackbone()
        else:
            print(f'{arch} not found using Resnet')
            self.backbone = ResBackbone()

        self.hm = nn.Sequential(
            nn.Conv2d(K * 64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * 64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.Dropout2d(p=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2*K, kernel_size=1, stride=1, padding=0, bias=True),
            )
        _fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(64, head_conv,kernel_size=3, padding=1, bias=True),
            nn.Dropout2d(p=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
            )
        _fill_fc_weights(self.wh)

    def forward(self, x):
        # x => (batch_size, c, t, h, w)
        x = [self.backbone(x[:, :, i, :, :]) for i in range(self.K)] # [(-1, 64, 48, 64)] * K
        output_wh = torch.cat([self.wh(feature) for feature in x], dim=1) # (-1, 2*K, 48, 64)
        x = torch.cat(x, dim=1)
        return {
            'hm' : self.hm(x), # (-1, num_classes, 48, 64)
            'mov': self.mov(x), # (-1, 2*K, 48, 64)
            'wh' : output_wh, # (-1, 2*K, 48, 64)
        }

class MOC_Net_PL(MOC_Net):
    '''
    arch => resnet | mnasnet | resnet_skip
    '''
    def __init__(self, arch, num_classes, K, head_conv=256,
                 hm_lambda=1, wh_lambda=1, mov_lambda=0.1, loss_type='focal',
                 **kwargs):
        self.save_hyperparameters()
        super().__init__(
                        arch=arch,
                        num_classes=num_classes,
                        K=K,
                        head_conv=head_conv,
        )
        
if __name__ == '__main__':
    from torchinfo import summary
    x = torch.randn(1, 3, 7, 192, 256)
    model = MOC_Net_PL(arch='resnet', num_classes=24, K=7) #65.3MB
    model = MOC_Net_PL(arch='mnasnet', num_classes=24, K=7) #40.09MB
    model = MOC_Net_PL(arch='resnet_skip', num_classes=24, K=7) #66.78MB
    y = model(x)
    for k in y.keys():
        print(k, y[k].shape)
    summary(model, x.shape)