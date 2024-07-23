from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch

from src.utils.losses import WeightedBCELoss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def weighted_bce_loss(probs, targets, weight=torch.tensor([4.,1.],dtype=torch.float32)):
        loss = - (weight[1] * targets * torch.log(probs + 1e-6) + weight[0] * (1 - targets) * torch.log(1 - probs + 1e-6))
        return loss.mean()


class BasicTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        
        self.criterion = WeightedBCELoss(weight_true=[5.,0.3125]).to(device) #real/fake classification loss
        self.class_loss = nn.CrossEntropyLoss(weight=torch.tensor([20.,1.05,1.],dtype=torch.float32)).to(device) #people num classifcation
        self.noise_loss = weighted_bce_loss #true/false noise binary classification
        
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(inputs['data']) #[batch, 2] real / fake binary classification
        label = inputs['label'] #[batch,2] real / fake binary classification
        people_cnt = inputs['people_cnt'] #[batch, 3] 0 1 2 people num classification
        is_noise = inputs['is_noise'] #[batch,1] true/false binary classification for predict whether noise exist
                
        loss_1 = self.criterion(output[:,:2],label) #bce loss in baseline
        loss_2 = self.class_loss(output[:,2:5],people_cnt.view(-1)) #target은 [batch] 차원
        loss_3 = self.noise_loss(output[:,5:],is_noise) #둘다 [batch,1] 차원
        
        loss = loss_1 + 0.01 * loss_2 + 0.01 * loss_3 #multitask learning
                
        if return_outputs:
            return loss, output[:,:2], label
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,True)
        
        return (eval_loss,pred,label)