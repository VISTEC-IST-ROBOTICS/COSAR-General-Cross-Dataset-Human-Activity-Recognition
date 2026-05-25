# learner.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from torchmetrics.classification import MulticlassF1Score

from abc import ABC, abstractmethod
import numpy as np
import time

from models import Baseline, BAMS, CoSAR, LLM4HAR
from utils.utils import print_head_border, PrintProgressBar, get_confusion_matrix, seed_worker

# from torchsummary import summary
from torchinfo import summary

np.set_printoptions(precision=4, suppress=True)

# create an abstract base class for learning (includes create_model, train, test)
class Learning(ABC):
    def __init__(self, num_classes, verbose=False, seed:int=42, **kwargs):
        self.net    = None
        self.num_classes = num_classes
        self.verbose    = verbose
        self.kwargs     = kwargs
        
        # set the device of operation following device defined
        if 'cuda' in self.kwargs['device']:
            # get the device name
            device_name = self.kwargs['device']
            
            # save the device
            self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
            
            # set the number of gpu devices
            device_num = int(device_name[-1])
            torch.cuda.set_device(device_num)
        else:
            self.device = torch.device('cpu')
    
    @abstractmethod
    def create_model(self):
        pass
    
    def initialize_model(self, params=None):
        self.net = self.create_model()
        if params is not None:
            self.net.load_state_dict(deepcopy(params))
    
    @abstractmethod
    def train(self, 
              train_dataset, 
              validation_dataset=None, 
              batch_size:int=128,
              epochs:int=90,
              lr:float=1e-4,
              weight_decay:float=0.1,
              params=None, 
              stage='pretrain',
        ):
        pass
    
    @abstractmethod
    def train_on_batch(self, data_loader, optimizer, num_update=1):
        pass
            
    # Helper methods (no abstract decorator)
    def setup_dataloaders(self, train_dataset, batch_size, stage='pretrain', validation_dataset=None, ):
        if stage == 'pretrain':
            loader_config = {
                'num_workers': 4,
                'pin_memory': True,
                'prefetch_factor': 4,
                'persistent_workers': True
            }
        elif stage == 'finetune':
            loader_config = {}
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                                  **loader_config)

        if validation_dataset is not None:
            validate_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)
        else:
            validate_loader = None
            
        return train_loader, validate_loader

    def test(self, test_dataset, name_classes, params=None, confusion_matrix=False):
        
        # create validation loader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=len(test_dataset),
            shuffle=False
        )
        
        # check the model for evaluation
        if self.net is None:
            self.net = self.create_model()
            # params is needed to load
            if params is None:
                raise ValueError("Model parameters must be provided for testing.")
            self.net.load_state_dict(deepcopy(params))
        else:
            # load params if provided
            if params is not None:
                self.net.load_state_dict(deepcopy(params))
            
        # create y prediction stroring for evaluation, and move to cpu
        y_preds = torch.empty(len(test_dataset), dtype=torch.long).to('cpu')
        # create evaluation matrices
        f1_scores, f1_matrics = self.create_f1_matrix()
        
        # start evaluation
        self.net.eval()
        with torch.no_grad():
            for _, (data, target) in enumerate(test_loader):
                data, target    = data.to(self.device), target.to(self.device)
                
                output  = self.net(data, test=True)
                loss    = F.cross_entropy(output, target)
                
                y_preds = torch.argmax(output, dim=1).to('cpu')
                y_true  = target.to('cpu')
                
                # probs = torch.softmax(output, dim=-1)
                # probs = probs.detach().cpu().numpy()
                        
        # collect f1 scores
        for matric_name in f1_matrics.keys():
            f1_matrics[matric_name].update(y_preds.to(self.device), y_true.to(self.device))
            f1_scores[matric_name] = f1_matrics[matric_name].compute()
        
        # collect accuracy score
        accuracy = ((y_preds == y_true).sum().item() * 100) / len(y_true)
        f1_weight = f1_scores['weighted'] * 100
        
        # plot confusion matrix if required
        if confusion_matrix:
            _ = get_confusion_matrix(y_pred=y_preds, y_true=y_true, tasks=name_classes)
        
        self.y_preds_finish = y_preds
        self.y_trues_finish = y_true
  
        return accuracy, f1_weight, loss.item()

    def get_prediction(self):
        return self.y_true, self.y_pred
            
    def create_f1_matrix(self):
        f1_scores   = {}
        f1_matrics  = {
            'weighted': MulticlassF1Score(num_classes=self.num_classes, average='weighted').to(self.device),
            'macro': MulticlassF1Score(num_classes=self.num_classes, average='macro').to(self.device),
            'each': MulticlassF1Score(num_classes=self.num_classes, average=None).to(self.device),
        }
        return f1_scores, f1_matrics
        
    def save_model(self, path, filename:str=''):
        """
        The method saves the model parameters to the specified path.
        path: str       - The directory path where the model will be saved.
        filename: str   - The name of the file to save the model parameters.
        """
        import os
        
        if filename == '':
            filename = 'model.pth'
        
        file_path = os.path.join(path, 'trained_model', filename)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.net.state_dict(), file_path)
        
        print(f'Model saved to {file_path}')
    
    def calculate_class_weights(self, dataset):
        """Calculate class weights for imbalanced dataset
        https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
        """
        from collections import Counter
        
        # Get all labels
        labels = []
        for _, label in dataset:
            labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        
        # Count classes
        class_counts = Counter(labels)
        total_samples   = len(labels)
        num_classes     = len(class_counts)
        
        # Calculate weights
        weights = []
        for i in range(num_classes):
            weight = total_samples / (num_classes * class_counts.get(i, 1))
            weights.append(weight)
        
        weights = torch.FloatTensor(weights).to(self.device)
        return weights  

class SupervisedLearning(Learning):
    def __init__(self, num_classes:int, feature_size, model_config, model_name:str='cosar', verbose=False, **kwargs):
        super().__init__(num_classes, verbose, **kwargs)
        self.model_config   = model_config
        self.num_classes    = num_classes
        self.feature_size   = feature_size
        self.model_name     = model_name
        self.verbose        = verbose
        self.kwargs         = kwargs
        self.log_freq       = 5
        
    def create_model(self):
        if self.model_name == 'baseline':
            net = Baseline(**self.model_config)
        elif self.model_name == 'bams':
            net = BAMS(**self.model_config)
        elif self.model_name == 'llm4har':
            net = LLM4HAR(**self.model_config)
            print(f"num_layers: {sum(1 for _ in net.knowledge.gpt2.h)}")
        else:
            net = CoSAR(feature_size=(self.feature_size[-2], self.feature_size[-1]), **self.model_config)
        return net.to(self.device)
    
    def train(self, 
              train_dataset, 
              validation_dataset=None, 
              batch_size:int=128,
              epochs:int=90,
              lr:float=1e-4,
              weight_decay:float=0.1,
              params=None, 
              stage='pretrain',
        ):

        # use helper to setup dataloaders
        self.initialize_model()
        # setup dataloaders
        train_loader, validate_loader = self.setup_dataloaders(
            train_dataset, 
            batch_size, stage=stage, validation_dataset=validation_dataset
        )
        # create optimizer
        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.AdamW(
        #     filter(lambda p: p.requires_grad, self.net.parameters()),
        #     lr=lr,
        #     weight_decay=weight_decay
        # )
        # # create scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(epochs/2),         # First restart after half of epochs
        )
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print('Optimizer:', optimizer)
        print('Scheduler:', scheduler)
        
        print_head_border('Start Training')
        
        self.current_lr = lr
        self.start_training_time = time.time()

        rng_state = torch.get_rng_state()
        self.net.eval()
        summary(self.net, input_size=(1, *train_dataset.X_train.shape[1:]))
        torch.set_rng_state(rng_state)          # restore exactly
        self.net.train()
        
        for cur_iter in range(epochs):
            if (cur_iter+1) % self.log_freq == 0 or cur_iter == 0:
                self.progress_bar = PrintProgressBar(len(train_loader),
                                                total_iter=epochs,
                                                step=len(train_loader),
                                                prefix='',
                                                suffix='Completed',
                                                length=15)
                
            # train on batch
            self.current_iter = cur_iter
            self.train_on_batch(train_loader, optimizer)
            
            total_time = time.time() - self.start_training_time
            
            # validate model
            if validation_dataset is not None:
                validate_acc, validate_loss = self.validate(validate_loader)
                # update progress bar with validation results
                if (cur_iter+1) % self.log_freq == 0 or cur_iter == 0 or (cur_iter+1) == epochs:
                    self.progress_bar(
                        cur_iter+1,
                        len(train_loader),
                        total_time,
                        validate_acc,
                        validate_loss,
                        lr=self.current_lr,
                    )
            
            # step the scheduler
            scheduler.step()
            # update current lr
            self.current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            
                
        print('***********************************************')
        print('**                 Finished                  **')
        print('***********************************************')
                
    def train_on_batch(self, data_loader, optimizer, num_update=1):
        self.net.train()
        
        for step, (x, y) in enumerate(data_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # forward pass
            for _ in range(num_update):
                y_pred = self.net(x, train=True)
                loss   = F.cross_entropy(y_pred, y)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                
                # calculate training time, accuracy and loss
                training_time = time.time() - self.start_training_time
                acc  = ((y_pred.argmax(dim=1) == y).sum().item() * 100) / len(y)
                acc  = round(acc, 2)
                loss = round(loss.item(), 4)
            
            # update pregress bar
            if (self.current_iter+1) % self.log_freq == 0 or self.current_iter == 0:
                self.progress_bar(self.current_iter+1, step, training_time, acc, loss)
                
    def validate(self, data_loader, visualize=False) -> tuple:
        """ Validate the model on the validation dataset """
        
        validation_acc = []
        validation_loss = [] 
        
        self.net.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y    = x.to(self.device), y.to(self.device) 
                y_pred  = self.net(x, test=True)

                # Get the performance of the model
                val_loss        = F.cross_entropy(y_pred, y)
                _, prediction   = torch.max(y_pred, 1)
                accuracy    = (torch.sum(prediction == y).item() / len(y)) * 100
                
                validation_loss.append(val_loss.item())
                validation_acc.append(accuracy)
        
        validation_acc = round(np.mean(validation_acc), 3)
        validation_loss = round(np.mean(validation_loss), 4)
        
        return validation_acc, validation_loss
        
def create_leanner(num_classes, feature_size, device='cpu', model='cosar', learning='supervised', seed:int=32, **kwargs):
    """
    Factory function to create learning instances
    
    Args:
        method: Training method
        model: Model architecture ('bams', etc.)
        learning: Learning strategy ('supervised', 'finetune', etc.)
    """
    # Model-specific configurations
    MODEL_CONFIGS = {
        'bams': {
            'num_sensors': 2,
            'num_channels': 1,
        },
        'cosar': {
            'num_sensors': 2,
            'num_channels': 1,
        },
        'baseline':{},
        'llm4har': {
            'in_channels': 6,       # n_sensors * n_channels = 2 * 3
            'T':           150,     # time 
            'segment_len': 50,      # T must be divisible by segment_len → 150/15=10
            'd_model':     128,
            'num_layers':  4,
            'num_classes': num_classes
        },
    }
    # Learner classes mapping
    LEARNERS = {
        'supervised': SupervisedLearning,
        # 'finetune': FineTuneLearning,
        # 'transfer': TransferLearning,
    }
    # Build complete model config (base + model-specific)

    model_config = {
        'num_classes': num_classes,  # Always included
        **MODEL_CONFIGS.get(model, {})  # Add model-specific params
    }
    
    # Validate learning strategy
    if learning not in LEARNERS:
        raise ValueError(f"Unknown learning '{learning}'. Choose from: {list(LEARNERS.keys())}")

    # create and return learner
    return LEARNERS[learning](
        num_classes=num_classes,
        model_config=model_config,
        # baseline=kwargs.get('baseline', arguments.baseline),
        model_name=model,
        feature_size=feature_size,
        device=device,
        verbose=kwargs.get('verbose', True),
        seed=seed
    )
