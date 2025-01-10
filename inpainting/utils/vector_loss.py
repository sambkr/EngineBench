import torch
from utils.evaluation import RI, MI

class VectorLoss(torch.nn.Module):
    def __init__(self, loss_type: str, alpha1=0.5, alpha2=0.001):
        """
        Args:
            loss_type (str): Type of loss ('RI', 'MI', 'RI_MI', or 'comb').
            alpha1 (float): Weighting for RI and MI in RI_MI loss.
            alpha2 (float): Weighting for the combination of vector loss and MSE.
        """
        super(VectorLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mse_loss = torch.nn.MSELoss()
        self.loss_map = {
            'RI': self.RI_loss,
            'MI': self.MI_loss,
            'RI_MI': self.RI_MI_loss,
            'comb': self.comb_loss
        }

    @staticmethod
    def RI_loss(A,B):
        return (1 - RI(A,B))/2
    
    @staticmethod
    def MI_loss(A,B):
        return 1 - MI(A,B)
    
    def RI_MI_loss(self,A,B):
        return self.alpha1*self.RI_loss(A,B) + (1-self.alpha1)*self.MI_loss(A,B)
    
    def comb_loss(self,A,B):
        return self.alpha2*self.RI_MI_loss(A,B) + (1-self.alpha2)*self.mse_loss(A,B)
    
    def forward(self,A,B):
        if self.loss_type not in self.loss_map:
            raise ValueError(f'Invalid loss type: {self.loss_type}. Must be RI, MI, RI_MI, or comb')
        
        return self.loss_map[self.loss_type](A, B)        