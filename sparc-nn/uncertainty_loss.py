import torch
import torch.nn as nn

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_losses):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_dict):
        total_loss = 0
        
        for i, (key, raw_loss) in enumerate(loss_dict.items()):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            
            weighted_loss = 0.5 * precision * raw_loss + 0.5 * log_var
            
            total_loss += weighted_loss
            
        return total_loss