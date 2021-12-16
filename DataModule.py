import pytorch_lightning as pl
import torch
import os
from dataset import UCFDataset

class VideoDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 K=7, 
                 down_ratio=4,
                 spatial_resolution=[192, 256],
                 mean=[0.40789654, 0.44719302, 0.47026115],
                 std=[0.28863828, 0.27408164, 0.27809835], 
                 batch_size=1, 
                 num_workers=None, 
                 pin_memory=True):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size 
        self.num_workers = os.cpu_count() - 1 if num_workers is None else num_workers
        self.pin_memory = pin_memory

        self.num_classes = self.Dataset.num_classes
        self.K = K
        self.down_ratio = down_ratio
        self.spatial_resolution = spatial_resolution
        self.mean = mean
        self.std = std 

    def setup(self, stage=None):
        self.train_ds = UCFDataset(root_dir=self.root_dir,
                            mode='train',
                            down_ratio=self.down_ratio,
                            K=self.K, 
                            spatial_resolution=self.spatial_resolution,
                            mean=self.mean,
                            std=self.std,
                        )
        self.val_ds = UCFDataset(root_dir=self.root_dir,
                            mode='val',
                            down_ratio=self.down_ratio,
                            K=self.K, 
                            spatial_resolution=self.spatial_resolution,
                            mean=self.mean,
                            std=self.std,
                        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.train_ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )
if __name__ == '__main__':
    datamodule = VideoDataModule(
                    root_dir='ucf24',
                    K=7,
                    down_ratio=4,
                    spatial_resolution=[192, 256],
                    mean=[0.40789654, 0.44719302, 0.47026115],
                    std=[0.28863828, 0.27408164, 0.27809835], 
                    batch_size=2,
                    num_workers=0,
                    pin_memory=True,
                    )
    print("Number of classes ", datamodule.num_classes)

    train_dl = datamodule.train_dataloader()
    print("Len of train_dl", len(train_dl))

    val_dl = datamodule.val_dataloader()
    print("Len of val_dl", len(val_dl))

    for data in train_dl:
        break

    print(data.keys()) # 'input', 'hm', 'mov', 'wh', 'mask', 'index', 'index_all']
    print(data['hm'].shape)
    print(data['mov'].shape)
    print(data['wh'].shape)