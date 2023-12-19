from typing import Dict, Any, Type

import torch
from torch import nn

import pytorch_lightning as pl

import copy

from model import VIME
from misc.scorer import BaseScorer

class PLVIME(pl.LightningModule):
    """The pytorch lightning module of VIME
    """
    def __init__(self,
                 model_hparams: Dict[str, Any],
                 optim: torch.optim,
                 optim_hparams: Dict[str, Any],
                 scheduler: torch.optim.lr_scheduler,
                 scheduler_hparams: Dict[str, Any],
                 num_categoricals: int,
                 num_continuous: int,
                 u_label,
                 loss_fn: nn.Module,
                 scorer: Type[BaseScorer],
                 random_seed: int = 0,
    ) -> None:
        """Initialize the pytorch lightining module of VIME

        Args:
            model_hparams (Dict[str, Any]): The hyperparameters of VIME
            optim (torch.optim): The optimizer for training
            optim_hparams (Dict[str, Any]): The hyperparameters of the optimizer
            scheduler (torch.optim.lr_scheduler): The scheduler for training
            scheduler_hparams (Dict[str, Any]): The hyperparameters of the scheduler
            num_categoricals (int): The number of categorical features
            num_continuous (int): The number of continuous features
            u_label (Any): The specifier for unlabeled data.
            loss_fn (nn.Module): The loss function of pytorch
            scorer (BaseScorer): The scorer to measure the performance
            random_seed (int, optional): The random seed. Defaults to 0.
        """
        super().__init__()

        pl.seed_everything(random_seed)

        hparams = copy.deepcopy(model_hparams)
        self.alpha1 = hparams["alpha1"]
        self.alpha2 = hparams["alpha2"]
        del hparams["alpha1"]
        del hparams["alpha2"]
        
        self.beta = hparams["beta"]
        del hparams["beta"]
        
        self.K = hparams["K"]
        self.consistency_len = self.K + 1
        del hparams["K"]
        
        
        self.model = VIME(**hparams)

        self.optim = getattr(torch.optim, optim)
        self.optim_hparams = optim_hparams

        self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)
        self.scheduler_hparams = scheduler_hparams
        
        self.num_categoricals = num_categoricals
        self.num_continuous = num_continuous
        
        self.u_label = u_label
        
        self.pretraining_mask_loss = nn.BCELoss()
        self.pretraining_feature_loss1 = nn.CrossEntropyLoss()
        self.pretraining_feature_loss2 = nn.MSELoss()
        
        self.consistency_loss = nn.MSELoss()
        self.loss_fn = loss_fn()
        
        self.scorer = scorer
            
        self.do_pretraining()

        self.pretraining_step_outputs = []
        self.finetunning_step_outputs = []
        self.save_hyperparameters()

        

    def configure_optimizers(self):
        """Configure the optimizer
        """
        self.optimizer = self.optim(self.parameters(), **self.optim_hparams)
        if len(self.scheduler_hparams) == 0:
            return [self.optimizer]
        self.scheduler = self.scheduler(self.optimizer, **self.scheduler_hparams)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'} ]

    def do_pretraining(self) -> None:
        """Set the module to pretraining
        """
        self.model.do_pretraining()
        self.training_step = self.pretraining_step
        self.on_validation_start = self.on_pretraining_validation_start
        self.validation_step = self.pretraining_step
        self.on_validation_epoch_end = self.pretraining_validation_epoch_end

    def do_finetunning(self) -> None:
        """Set the module to finetunning
        """
        self.model.do_finetunning()
        self.training_step = self.finetuning_step
        self.on_validation_start = self.on_finetunning_validation_start
        self.validation_step = self.finetuning_step
        self.on_validation_epoch_end = self.finetuning_validation_epoch_end

    def forward(self,
                batch:Dict[str, Any]
    ) -> torch.FloatTensor:
        """Do forward pass for given input

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The output of forward pass
        """
        return self.model(batch)
    

    def get_pretraining_loss(self, batch:Dict[str, Any]):
        """Calculate the pretraining loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of pretraining step
        """
        mask_output, feature_output = self.model.pretraining_step(batch["input"])
        
        mask_loss = self.pretraining_mask_loss(mask_output, batch["label"][0])
        feature_loss1, feature_loss2 = 0, 0
        if self.num_categoricals > 0:
            feature_loss1 = self.pretraining_feature_loss1(feature_output[:, :self.num_categoricals], batch["label"][1][:, :self.num_categoricals])
        if self.num_continuous > 0:
            feature_loss2 = self.pretraining_feature_loss2(feature_output[:, self.num_categoricals:], batch["label"][1][:, self.num_categoricals:])
        final_loss = mask_loss + self.alpha1 * feature_loss1 + self.alpha2 * feature_loss2

        return final_loss
    
    def pretraining_step(self,
                      batch,
                      batch_idx: int
    ) -> Dict[str, Any]:
        """Pretraining step of VIME

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            Dict[str, Any]: The loss of the pretraining step
        """

        loss = self.get_pretraining_loss(batch)
        self.pretraining_step_outputs.append({
            "loss" : loss
        })
        return {
            "loss" : loss
        }

    def on_pretraining_validation_start(self):
        """Log the training loss of the pretraining
        """
        if len(self.pretraining_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.pretraining_step_outputs]).cpu().mean()
            
            self.log("train_loss", train_loss, prog_bar = True)
            
            self.pretraining_step_outputs = []    
        return super().on_validation_start() 
    
    def pretraining_validation_epoch_end(self) -> None:
        """Log the validation loss of the pretraining
        """
        val_loss = torch.Tensor([out["loss"] for out in self.pretraining_step_outputs]).cpu().mean()

        self.log("val_loss", val_loss, prog_bar = True)
        self.pretraining_step_outputs = []
        return super().on_validation_epoch_end()


    def get_finetunning_loss(self, batch:Dict[str, Any]):
        """Calculate the finetunning loss

        Args:
            batch (Dict[str, Any]): The input batch

        Returns:
            torch.FloatTensor: The final loss of finetunning step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        x = batch["input"]
        y = batch["label"]
        
        unsupervised_loss = 0
        unlabeled = x[y == self.u_label]

        if len(unlabeled) > 0:
            u_y_hat = self.model.finetunning_step(unlabeled)
            target = u_y_hat[::self.consistency_len]
            target = target.repeat(1, self.K).reshape((-1, u_y_hat.shape[-1]))
            preds = torch.stack([u_y_hat[i, :] for i in range(len(u_y_hat)) if i % self.consistency_len != 0], dim = 0)
            unsupervised_loss += self.consistency_loss(preds, target)
        
        labeled_x = x[y != self.u_label].squeeze()
        labeled_y = y[y != self.u_label].squeeze()

        y_hat = self.model.finetunning_step(labeled_x).squeeze()

        supervised_loss = self.loss_fn(y_hat, labeled_y)
        
        loss = supervised_loss + self.beta * unsupervised_loss
        
        return loss, labeled_y, y_hat
        
    
    def finetuning_step(self,
                      batch,
                      batch_idx: int
    ) -> Dict[str, Any]:
        """Finetunning step of VIME

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            Dict[str, Any]: The loss of the finetunning step
        """
        loss, y, y_hat = self.get_finetunning_loss(batch)
        self.finetunning_step_outputs.append(
            {
            "loss" : loss,
            "y" : y,
            "y_hat" : y_hat
        }
        )
        return {
            "loss" : loss
        }
    
    def on_finetunning_validation_start(self):
        """Log the training loss and the performance of the finetunning
        """
        if len(self.finetunning_step_outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.finetunning_step_outputs]).cpu().mean()
            y = torch.cat([out["y"] for out in self.finetunning_step_outputs]).cpu().detach().numpy()
            y_hat = torch.cat([out["y_hat"] for out in self.finetunning_step_outputs]).cpu().detach().numpy()
            
            train_score = self.scorer(y, y_hat)
            
            self.log("train_loss", train_loss, prog_bar = True)
            self.log("train_" + self.scorer.__name__, train_score, prog_bar = True)
            self.finetunning_step_outputs = []   
            
        return super().on_validation_start()
    
    def finetuning_validation_epoch_end(self) -> None:
        """Log the validation loss and the performance of the finetunning
        """
        val_loss = torch.Tensor([out["loss"] for out in self.finetunning_step_outputs]).cpu().mean()

        y = torch.cat([out["y"] for out in self.finetunning_step_outputs]).cpu().numpy()
        y_hat = torch.cat([out["y_hat"] for out in self.finetunning_step_outputs]).cpu().numpy()
        val_score = self.scorer(y, y_hat)

        self.log("val_" + self.scorer.__name__, val_score, prog_bar = True)
        self.log("val_loss", val_loss, prog_bar = True)
        self.finetunning_step_outputs = []      
        return super().on_validation_epoch_end()
    
    
    def predict_step(self, batch, batch_idx: int
    ) -> torch.FloatTensor:
        """The perdict step of VIME

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        y_hat = self.model.finetunning_step(batch["input"])

        return y_hat