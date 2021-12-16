import torch.nn.functional as F
from torch import optim
from utils.losses import MOCLoss
import pytorch_lightning as pl

class VideoObjectDetectionModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Define the loss function
        self.loss = MOCLoss(
                        self.model.hparams.loss_type, 
                        self.model.hparams.hm_lambda, 
                        self.model.hparams.wh_lambda, 
                        self.model.hparams.mov_lambda
                        )

    def configure_optimizers(self):
        if self.model.hparams.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), self.model.hparams.lr, momentum = 0.9)
        
        elif self.model.hparams.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), self.model.hparams.lr)
        
        elif self.model.hparams.optimizer == 'adamax':
            return optim.Adamax(self.model.parameters(), self.model.hparams.lr)
    
    def run_epoch(self, phase, batch, batch_idx):
        assert batch['video'].shape[2] == self.model.K
        
        output = self.model(batch['video'])
        loss, loss_stats = self.loss(output, batch)

        self.log_dict(
            {
                f'{phase}_loss' : loss_stats['loss'], 
                f'{phase}_loss_hm' : loss_stats['loss_hm'], 
                f'{phase}_loss_mov' : loss_stats['loss_mov'], 
                f'{phase}_loss_wh' : loss_stats['loss_wh'], 
                f'{phase}_dice_loss' : loss_stats['dice_loss'], 
            },
            on_epoch = True,
            prog_bar = True,
        )

        return loss.mean()

    def training_step(self, batch, batch_idx):
        return self.run_epoch("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.run_epoch("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.run_epoch("test", batch, batch_idx)
    