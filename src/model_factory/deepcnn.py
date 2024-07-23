import torch
import torch.nn as nn
import torchvision.models as models

class DEEPCNN(nn.Module):
    def __init__(self, args, num_classes=2, pretrained=True):
        super(DEEPCNN, self).__init__()
        self.model_name = args.model_name
        
        if self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes + 4) # +3 people classification, +1 noise classification
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
    
    def forward(self, x):
        embedding = self.model(x) #[batch, 5]
        output_1 = torch.sigmoid(embedding[:,:2]) #[batch, 2], real fake detection
        output_2 = torch.softmax(embedding[:,2:5], dim=1) #[batch, 3] people num classification
        output_3 = torch.sigmoid(embedding[:,5:]) #[batch, 1] noise classifciation        
        output = torch.cat([output_1,output_2,output_3], dim=1) #[batch, 5]
        return output
    
    def sigmoid_smoothing(self, x, alpha=1.7):
        return 1 / (1 + torch.exp(-x / alpha))
    
    def inference_forward(self,x):
        embedding = self.model(x) #[batch, 5]
        output_1 = self.sigmoid_smoothing(embedding[:,:2]) #[batch, 2], real fake detection
        output_2 = torch.softmax(embedding[:,2:5], dim=1) #[batch, 3] people num classification
        output_3 = torch.sigmoid(embedding[:,5:]) #[batch, 1] noise classifciation        
        output = torch.cat([output_1,output_2,output_3], dim=1) #[batch, 5]
        return output