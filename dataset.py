import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# 이 코드의 단점 : num_points가 고정되어있어야 함 

class InputDataset(Dataset):
    def __init__(self, root_dir, focal_length, focal_const, num_points):
        super(InputDataset, self).__init__()
        self.root_dir = root_dir
        self.x_dir = os.path.join(root_dir, 'world')
        self.y_dir = os.path.join(root_dir, 'image')
        self.label_dir = os.path.join(root_dir, 'pose.csv')
        self.focal_length = focal_length
        self.focal_const = focal_const
        self.num_points = num_points
        
        # load label ; R, t
        self.label = pd.read_csv(self.label_dir, header=None).to_numpy()
        self.r_df = self.label[:, :4]
        self.t_df = self.label[:, 4:]

        # load 2D/3D points 
        self.data = []
        for i in range(0, len(self.label)):
            self.x_files = os.path.join(self.x_dir, 'world_{}.csv'.format(i+1))
            x = pd.read_csv(self.x_files, header=None)
            x = x / focal_length * focal_const
            self.y_files = os.path.join(self.y_dir, 'image_{}.csv'.format(i+1))
            y = pd.read_csv(self.y_files, header=None)
            cor = pd.concat([x, y], axis=1)
            cor = np.array(cor.values).reshape(-1) # flatten
            cor = cor.reshape(-1, len(cor)) 
            self.data.append(cor)
        self.data = np.array(self.data)

        
    def __getitem__(self, index):
        corr = np.array(self.data[index], dtype=np.float32) # flattened correspondences
        R = np.array(self.r_df[index,:], dtype=np.float32) # Rotation
        t = np.array(self.t_df[index,:], dtype=np.float32) # translation
        return (corr, R, t)
        
    def __len__(self):
        return len(self.data)