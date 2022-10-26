from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from climatenet.base.base_model import BaseModel
from climatenet.utils.helpers import initialize_weights, set_trainable
from itertools import chain
from climatenet.utils.utils import Config
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import xarray as xr
from climatenet.utils.utils import Config
from os import path
import pathlib
from torchvision.models import resnet101

class UperNet():
    def __init__(self, config: Config = None, model_path: str = None):
    
        if config is not None and model_path is not None:
            raise ValueError('''Config and weight path set at the same time. 
            Pass a config if you want to create a new model, 
            and a weight_path if you want to load an existing model.''')

        if config is not None:
            # Create new model
            self.config = config
            self.network = UperNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
        elif model_path is not None:
            # Load model
            self.config = Config(path.join(model_path, 'config.json'))
            self.network = UperNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
            self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
        else:
            raise ValueError('''You need to specify either a config or a model path.''')

        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)        
        
    def train(self, dataset: ClimateDatasetLabeled):
        '''Train the network on the given dataset for the given amount of epochs'''
        self.network.train()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.train_batch_size, collate_fn=collate, num_workers=4, shuffle=True)
        for epoch in range(1, self.config.epochs+1):

            print(f'Epoch {epoch}:')
            epoch_loader = tqdm(loader)
            aggregate_cm = np.zeros((3,3))

            for features, labels in epoch_loader:
        
                # Push data on GPU and pass forward
                features = torch.tensor(features.values).cuda()
                labels = torch.tensor(labels.values).cuda()
                
                outputs = torch.softmax(self.network(features), 1)

                # Update training CM
                predictions = torch.max(outputs, 1)[1]
                aggregate_cm += get_cm(predictions, labels, 3)

                # Pass backward
                loss = jaccard_loss(outputs, labels)
                epoch_loader.set_description(f'Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

            print('Epoch stats:')
            print(aggregate_cm)
            ious = get_iou_perClass(aggregate_cm)
            print('IOUs: ', ious, ', mean: ', ious.mean())

    def predict(self, dataset: ClimateDataset, save_dir: str = None):
        '''Make predictions for the given dataset and return them as xr.DataArray'''
        self.network.eval()
        collate = ClimateDataset.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate)
        epoch_loader = tqdm(loader)

        predictions = []
        for batch in epoch_loader:
            features = torch.tensor(batch.values).cuda()
        
            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            preds = torch.max(outputs, 1)[1].cpu().numpy()

            coords = batch.coords
            del coords['variable']
            
            dims = [dim for dim in batch.dims if dim != "variable"]
            
            predictions.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))

        return xr.concat(predictions, dim='time')

    def evaluate(self, dataset: ClimateDatasetLabeled):
        '''Evaluate on a dataset and return statistics'''
        self.network.eval()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=4)

        epoch_loader = tqdm(loader)
        aggregate_cm = np.zeros((3,3))

        for features, labels in epoch_loader:
        
            features = torch.tensor(features.values).cuda()
            labels = torch.tensor(labels.values).cuda()
                
            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            predictions = torch.max(outputs, 1)[1]
            aggregate_cm += get_cm(predictions, labels, 3)

        print('Evaluation stats:')
        print(aggregate_cm)
        ious = get_iou_perClass(aggregate_cm)
        print('IOUs: ', ious, ', mean: ', ious.mean())

    def save_model(self, save_path: str):
        '''
        Save model weights and config to a directory.
        '''
        # create save_path if it doesn't exist
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 

        # save weights and config
        self.config.save(path.join(save_path, 'config.json'))
        torch.save(self.network.state_dict(), path.join(save_path, 'weights.pth'))

    def load_model(self, model_path: str):
        '''
        Load a model. While this can easily be done using the normal constructor, this might make the code more readable - 
        we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
        '''
        self.config = Config(path.join(model_path, 'config.json'))
        self.network = UperNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
        self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))

class UperNetModule(BaseModel):
    # Implementing only the object path
    def __init__(self, classes, channels=3, backbone='resnet101', pretrained=True, use_aux=True, fpn_out=256, freeze_bn=False, freeze_backbone=FALSE, **_):
        super(UperNetModule, self).__init__()

        if backbone == 'resnet34' or backbone == 'resnet18':
            feature_channels = [64, 128, 256, 512]
        else:
            feature_channels = [256, 512, 1024, 2048]
        self.backbone = ResNet(channels, pretrained=pretrained)
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, classes, kernel_size=3, padding=1)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels+(out_channels * len(bin_sizes)), channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class ResNet(nn.Module):
    def __init__(self, channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained) # if this line crashes, you need to download the pretrained model .pth and put it in ~/.cache/torch/checkpoints/
        model = torch.load("/home/alexis/resnet101-5d3b4d8f.pth") 
        if not pretrained or channels != 3:
            self.initial = nn.Sequential(
                nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16: s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8: s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8: 
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x
