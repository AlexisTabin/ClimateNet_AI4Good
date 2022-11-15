import pathlib
from os import path

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from climatenet.models.cgnet.cgnet import CGNet
from climatenet.models.unet.unet import UNetResnet, UNet
from climatenet.models.upernet.upernet import UperNet
from climatenet.models.segnet.segnet import SegResNet, SegNet
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from climatenet.utils.utils import Config

MODELS = {
    'upernet': UperNet,
    'cgnet': CGNet,
    'unetresnet': UNetResnet,
    'unet': UNet,
    'segnetresnet': SegResNet,
    'segnet': SegNet
}

class Trainer():
    '''
    The high-level Trainer class. 
    This allows training and running models without interacting with PyTorch code.
    If you are looking for a higher degree of control over the training and inference,
    we suggest you directly use the Module class, which is a PyTorch nn.Module.

    Parameters
    ----------
    config : Config
        The model configuration.
    model_path : str
        Path to load the model and config from.

    Attributes
    ----------
    config : dict
        Stores the model config
    network : module for the model name
        Stores the actual model (nn.Module)
    optimizer : torch.optim.Optimizer
        Stores the optimizer we use for training the model
    '''

    def __init__(self, config: Config = None, model_name: str = None, model_path: str = None):
        if model_name is None :
            raise ValueError('model_name must be specified')

        if config is not None and model_path is not None:
            raise ValueError('''Config and weight path set at the same time. 
            Pass a config if you want to create a new model, 
            and a weight_path if you want to load an existing model.''')

        if config is not None:
            # Create new model
            self.config = config
            self.network = MODELS[model_name](classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
        elif model_path is not None:
            # Load model
            self.config = Config(path.join(model_path, 'config.json'))
            self.network = MODELS[model_name](classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
            self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
        else:
            raise ValueError('''You need to specify either a config or a model path.''')

        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)   
        self.scaler = GradScaler()     
        
    def train(self, dataset: ClimateDatasetLabeled):
        '''Train the network on the given dataset for the given amount of epochs'''
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        torch.cuda.empty_cache()
        gc.collect()
        self.network.train()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.train_batch_size, collate_fn=collate, num_workers=4, shuffle=True)
        for epoch in range(1, self.config.epochs+1):

            print(f'Epoch {epoch}:')
            epoch_loader = tqdm(loader)
            aggregate_cm = np.zeros((3,3))

            for features, labels in epoch_loader:
                self.optimizer.zero_grad()

                # Runs the forward pass with autocasting.
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Push data on GPU and pass forward
                    features = torch.tensor(features.values).cuda()
                    labels = torch.tensor(labels.values).cuda()
                    
                    outputs = torch.softmax(self.network(features), 1)
                    # Pass backward
                    loss = jaccard_loss(outputs, labels)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                self.scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                self.scaler.update()

                # Update training CM
                predictions = torch.max(outputs, 1)[1]
                aggregate_cm += get_cm(predictions, labels, 3)

                epoch_loader.set_description(f'Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()

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

    def load_model(self, model_path: str, model_name: str=None):
        '''
        Load a model. While this can easily be done using the normal constructor, this might make the code more readable - 
        we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
        '''
        if model_name is None:
            raise ValueError('model_name must be specified')
        self.config = Config(path.join(model_path, 'config.json'))
        self.network = MODELS[model_name](classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
        self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
