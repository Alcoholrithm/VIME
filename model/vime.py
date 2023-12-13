import torch.nn as nn

from .vime_self import VIMESelfSupervised
from .vime_semi import VIMESemiSupervised

class VIME(nn.Module):
    def __init__(self, 
                encoder_dim: int, predictor_hidden_dim: int, predictor_output_dim: int):
        super().__init__()
        
        self.self_sl = VIMESelfSupervised(encoder_dim)
        self.semi_sl = VIMESemiSupervised(encoder_dim, predictor_hidden_dim, predictor_output_dim)
        
        self.do_pretraining()
    
    def do_pretraining(self):
        self.forward = self.pretraining_step
    
    def do_finetuning(self):
        self.forward = self.finetunning_step
        
    def pretraining_step(self, x):
        mask_output, feature_output = self.self_sl(x)
        return mask_output, feature_output
    
    
    def finetunning_step(self, x):
        x = self.self_sl.h(x)
        logits = self.semi_sl(x)
        return logits