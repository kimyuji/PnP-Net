#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
#%%
class inter_loss(nn.Module):
    def __init__(self):
        super(inter_loss, self).__init__()
    
    def forward(self, r_pred, r_target, t_pred, t_target, lam = 1):
        # normalize by unit 
        r_error = torch.linalg.norm(r_target - r_pred) * lam # R과 t의 scale 비슷하게 맞추기 
        t_error = torch.linalg.norm(t_target - t_pred) 
        return r_error, t_error


class Trainer(object):
    def __init__(self, args, model, r_thres = 1, t_thres = 0.2, lam = 1):
        self.args = args
        self.model = model
        self.r_thres = r_thres
        self.t_thres = t_thres
        self.lam = lam

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = inter_loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters()) # lr, decay 


    def train(self, data_loader): 
        epoch_loss, succ, = 0, 0
        self.model.train() 

        for i, input in enumerate(data_loader): 
            data = input[0]
            r = input[1]
            t = input[2]

            r_pred, t_pred = self.model(data)
            r_loss, t_loss = self.criterion(r_pred, r, t_pred, t, self.lam)
            loss = r_loss + t_loss

            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            # succ += ((r_loss < self.r_thres) & (t_loss < self.t_thres)).sum()
            epoch_loss += loss.item() # i일때의 loss만

        return epoch_loss/len(data_loader) # success rate 제대로x 모든 데이터에 대해, iteration 한번 끝나면 evaluate 

    def evaluate(self, data_loader):
        epoch_loss, succ = 0, 0
        self.model.eval()

        with torch.no_grad():
            for input in data_loader:  
                data = input[0]
                r = input[1]
                t = input[2]

                r_pred, t_pred = self.model(data)
                r_loss, t_loss = self.criterion(r_pred, r, t_pred, t)
                loss = r_loss + t_loss

                succ += ((r_loss < self.r_thres) & (t_loss < self.t_thres)).sum()
                epoch_loss += loss.item()

        return epoch_loss/len(data_loader), succ / len(data_loader) * 100