import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

from models.model import MOC_Net_PL
from trainer import VideoObjectDetectionModule
from DataModule import VideoDataModule
from utils.utils import setup_dir

def launch_tensorboard(log_dir):
    setup_dir(log_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    webbrowser.open(url)

def parser_args():
  def nullable_str(s):
    if s.lower() in ['null', 'none', '']:
      return None
    return s
  
  def gpu_str(s):
    if s.lower() in ['null', 'none', '']:
      return -1
    else:
      try:
        output = int(s)
      except:
        s = s.strip()
        s = s.replace('[', '').replace(']', '')
        output = [int(x) for x in s.split(',')]
      return output

  parser = argparse.ArgumentParser()
  parser.add_argument('--arch', default='resnet', help='architecture of model => resnet | mnasnet | resnet_skip')
  parser.add_argument('--gpus', type=gpu_str, default=-1)
  parser.add_argument('--batch_size', type=int, default=-1)
  parser.add_argument('--num_workers', type=int, default=-1)
  parser.add_argument('--ckpt', type=nullable_str, help='pretrained checkpoint file', default='')
  
  return parser.parse_args()


def train(config, seed=137):
    pl.seed_everything(seed, workers=True)
    args = parser_args()
    print(args)

    if config['clear_log']:
        log_dir = os.path.join(config['default_root_dir'], 'lightning_logs')
        setup_dir(log_dir)

    checkpoint_cb = ModelCheckpoint(
        save_last=True,
        monitor='val_loss',
        mode='min',
        filename='{epoch}-{val_loss:.4f}',
    )
    datamodule = VideoDataModule(
        root_dir=config['dataset_dir'],
        K=7,
        down_ratio=4,
        spatial_resolution=[192, 256],
        batch_size=args.batch_size if args.batch_size != -1 else config['batch_size'],
        num_workers=args.num_workers if args.num_workers != -1 else config['num_workers'],
        pin_memory=True,
    )
    model = MOC_Net_PL(
                    arch=args.arch, 
                    num_classes=24, 
                    K = 7, 
                    lr=5e-4, 
                    optimizer='adam',
                    hm_lambda=1, wh_lambda=1, mov_lambda=0.1, loss_type='focal',
                    )
    model = VideoObjectDetectionModule(model)
    trainer = pl.Trainer(
                        max_epochs=config['max_epochs'], 
                        precision=config['precision'],
                        limit_train_batches=config['datasets_limits'][0],
                        limit_val_batches=config['datasets_limits'][1],
                        limit_test_batches=config['datasets_limits'][2],
                        default_root_dir=config['default_root_dir'],
                        callbacks=[checkpoint_cb],
                        gpus=args.gpus,
                        benchmark=True,
                        progress_bar_refresh_rate=config['progress_bar_refresh_rate'],
                        resume_from_checkpoint=args.ckpt, 
                        )
    trainer.fit(model, datamodule=datamodule)

########################################################
if __name__ == '__main__':
    from tensorboard import program
    import webbrowser

    local_config = {
        'clear_log': True,
        'dataset_dir' : 'ucf24',
        'datasets_limits' : (1.0, 1.0, 1.0), #(1, 1, 1),
        'max_epochs': 2,
        'batch_size' : 4, #8
        'num_workers' : 0,
        'precision' : 32,
        'default_root_dir' : '.',
        'progress_bar_refresh_rate' : None,
    }

    
    # launch_tensorboard(log_dir = os.path.join(local_config['default_root_dir'], 'lightning_logs'))
    train(local_config)
    