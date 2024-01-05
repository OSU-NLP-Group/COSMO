import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, ckpt_path='checkpoint.pt', best_ckpt_path='best_checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckpt_path = ckpt_path
        self.best_ckpt_path = best_ckpt_path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model, dense_optimizer, sparse_optimizer=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, dense_optimizer, sparse_optimizer)

            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving best ckpt ...')

            if sparse_optimizer:
                ckpt_dict = {'model': model.state_dict(), 'sparse_optimizer':sparse_optimizer.state_dict(), 'dense_optimizer':dense_optimizer.state_dict()}
            else:
                ckpt_dict = {'model': model.state_dict(), 'dense_optimizer':dense_optimizer.state_dict()}

            torch.save(ckpt_dict, self.best_ckpt_path)
            
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_checkpoint(val_loss, model, dense_optimizer, sparse_optimizer)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving best ckpt ...')

            if sparse_optimizer:
                ckpt_dict = {'model': model.state_dict(), 'sparse_optimizer':sparse_optimizer.state_dict(), 'dense_optimizer':dense_optimizer.state_dict()}
            else:
                ckpt_dict = {'model': model.state_dict(), 'dense_optimizer':dense_optimizer.state_dict()}

            torch.save(ckpt_dict, self.best_ckpt_path)

            self.save_checkpoint(val_loss, model, dense_optimizer, sparse_optimizer)

            self.val_loss_min = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, dense_optimizer, sparse_optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Saving ckpt ...')
        
        if sparse_optimizer:
            ckpt_dict = {'model': model.state_dict(), 'sparse_optimizer':sparse_optimizer.state_dict(), 'dense_optimizer':dense_optimizer.state_dict()}
        else:
            ckpt_dict = {'model': model.state_dict(), 'dense_optimizer':dense_optimizer.state_dict()}

        torch.save(ckpt_dict, self.ckpt_path)
        