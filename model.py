import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class Net(nn.Module):
    def __init__(self, num_points,  drop_percent=0.2):
        super(Net, self).__init__()
        self.num_points = num_points
        self.drop_percent = nn.Dropout(drop_percent)

        # deep neural network
        self.fc = nn.Sequential(
            nn.Linear(5*self.num_points, 20*self.num_points), 
            nn.ReLU(),
            nn.Linear(20*self.num_points, 5*self.num_points),
            nn.ReLU(),
            nn.Linear(5*self.num_points, 3*self.num_points),
            nn.ReLU(),
        )
        self.R_fc = nn.Sequential(
            nn.Linear(3*self.num_points, 2*self.num_points), 
            nn.ReLU(),
            nn.Linear(2*self.num_points, 2*self.num_points),
            nn.ReLU(),
            nn.Linear(2*self.num_points, 2*self.num_points),
            nn.ReLU(),
            nn.Linear(2*self.num_points, 2*self.num_points),
            nn.ReLU(),
            nn.Linear(2*self.num_points, 4),
            nn.ReLU(),
        )

        self.T_fc = nn.Sequential(
            nn.Linear(3*self.num_points, 2*self.num_points), 
            nn.ReLU(),
            nn.Linear(2*self.num_points, 2*self.num_points), 
            nn.ReLU(),
            nn.Linear(2*self.num_points, 2*self.num_points), 
            nn.ReLU(),
            nn.Linear(2*self.num_points, 2*self.num_points), 
            nn.ReLU(),
            nn.Linear(2*self.num_points, 3), 
            nn.ReLU(),
        )
        
        # LM layers can be implemented by "cv.solvePnP(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)"
    

    def forward(self, x, intrinsic):
        # ordering required
        x = self.fc(x)
        coarse_R = self.R_fc(x)
        coarse_T = self.T_fc(x)

        x = x.reshape(5,-1)
        pts3d = torch.index_select(x,dim=1,index=torch.LongTensor([0,1,2]))
        pts2d = torch.index_select(x,dim=1,index=torch.LongTensor([3,4]))
        # LM layers
        for i in range() :
            coarse_R, coarse_T = cv2.solvePnP(objectPoints=pts3d, imagePoints=pts2d, 
                    cameraMatrix=intrinsic, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, 
                    useExtrinsicGuess=True, rvec=coarse_R, tvec=coarse_T)
        return coarse_R, coarse_T