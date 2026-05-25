# utils folder
import os
import math
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from termcolor import colored
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


def calculate_sd_pooled(SDs:list, Ns:list):
    """Calculate the pooled standard deviation."""
    upper = 0
    for sd, n in zip(SDs, Ns):
        upper += (n - 1) * (sd ** 2)
    lower = sum(Ns) - len(Ns)
    return math.sqrt(upper / lower)

def get_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor,
    tasks: list = None,
    indent: str = "    ",
    acc: float = None,
    f1: float = None,
    normalize: str = 'true',
    verbose=True
):
    if tasks is None:
        raise ValueError("Class name list must be provided for confusion matrix.")

    f1_metric = MulticlassF1Score(num_classes=len(tasks), average='weighted')

    accuracy = 0.0
    f1_score_val = 0.0

    y_true = y_true.to('cpu')
    y_pred = y_pred.to('cpu')

    if acc:
        accuracy = (y_pred == y_true).sum().item() * 100 / len(y_true)

    if f1:
        f1_metric.update(y_pred, y_true)
        f1_score_val = f1_metric.compute().item() * 100

    y_true_np = y_true.view(-1).numpy()
    y_pred_np = y_pred.view(-1).numpy()

    conf_matrix = confusion_matrix(
        y_true_np, y_pred_np,
        labels=list(range(len(tasks))),  # ← always full shape
        normalize=normalize
    )

    df_cm = pd.DataFrame(conf_matrix, index=tasks, columns=tasks)

    if verbose:
        print(indent + f'Confusion matrix for {len(tasks)} tasks')
        print(indent + f"    ├── Unique values in y_pred: {torch.unique(y_pred).numpy()}")
        print(indent + f"    └── Unique values in y_true: {torch.unique(y_true).numpy()}")
        print(indent + df_cm.to_string().replace("\n", "\n" + indent))
        print(indent + '-' * 67)
        print(indent + f'Performance: Acc={accuracy:.4f}%, F1={f1_score_val:.4f}% from {len(y_true_np)} samples')
        print(indent + '-' * 67)

    return df_cm

def calculate_output_size(H_in, W_in, kernel_size, stride, padding=(0,0), dilation=(1,1)):
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
        
    # Calculate the output height
    H_out = math.floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    # Calculate the output width
    W_out = math.floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    
    return H_out, W_out

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # g = torch.Generator()
    # g.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    print(f"Worker {worker_id} seed: {worker_seed}")
    np.random.seed(worker_seed)
    random.seed(worker_seed)  # แก้ random_time_mask()

    
def print_head_border(context:str, length:int=80, symbol='*', color:str=None):
    border      = symbol * length
    side_border = symbol * 2

    num_retain = (length - 4) - len(context)
    total_context = (' '*(num_retain//2)) + context + (' '*(num_retain//2))
    print(colored(f'{border}\n{side_border}{total_context}{side_border}\n{border}', color=color))

class DicttoObj(object):
    def __init__(self, dictionary) -> None:
        super(DicttoObj, self).__init__()
        for key in dictionary:
            setattr(self, key, dictionary[key])
    
    def __str__(self) -> str:
        return str(self.__dict__)

class PrintProgressBar(object):
    def __init__(self, total, total_iter, step, prefix='', suffix='', decimals=1, length=20, printEnd="\r") -> None:
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        
        super(PrintProgressBar, self).__init__()
        self.total      = total
        self.total_iter = total_iter
        self.step       = step
        self.prefix     = prefix
        self.suffix     = suffix
        self.decimals   = decimals
        self.length     = length
        self.printEnd   = printEnd

        self.fill       = '█'
        self.non_fill   = '.'
        self.fr_start   = '['
        self.fr_end     = ']'
        # self.percents   = 100 * (iteration / float(total))

    def __call__(self, iter, step, ts, acc, loss, suffix=None, lr=None):
        ts = round(ts, 2)
        loss, acc = [round(val, 5) for val in (loss, acc)]

        percent = 100 * (step / float(self.total))
        filledLength = int(self.length * step // self.total)
        bar = self.fill * filledLength + self.non_fill * (self.length - filledLength)

        self.args = {'bar': bar, 'ts': ts, 'loss': loss, 'acc': acc, 'suffix': suffix, 'lr':lr}
        self.args = DicttoObj(self.args)

        # if val_acc != 0.0:
        #     print(self.__str__(iter, finished=True), end=self.printEnd)
        #     print()
        if step == self.step:
            print(self.__str__(iter, finished=True), end=self.printEnd)
            print()
        else:
            print(self.__str__(iter), end=self.printEnd)

    def __str__(self, iter, finished=False):
        if iter < 100:
            if iter < 10:
                iter = '  ' + str(iter)
            else:
                iter = ' ' + str(iter)
        else:
            iter = str(iter)
        
        #\r    {self.prefix} [{iter}/{self.total}]
        str_ =  f'{iter}/{self.total_iter} {self.fr_start}{self.args.bar}{self.fr_end} ETA: {self.args.ts:.2f}s' \
                f' - loss={self.args.loss:.4f}, acc={self.args.acc:.2f}%'
        if finished:
            return str_ + f' [last_lr={self.args.lr:.2e}]'
        else:
            return str_
        
def get_unique_filename(filepath):
    if not os.path.exists(filepath):
        return filepath
    
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"

