import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class DataModule_(pl.LightningDataModule):
    def __init__(self, path_train, batch_size):
        super(DataModule_, self).__init__()
        self.batch_size = batch_size
        self.path_train = path_train

    def setup(self, stage):
        self.dataset_train = ImageFolder(self.path_train)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.dataset, batch_size=self.batch_size)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.valid_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    dm = DataModule_(path_train='/home/dblab/sin/save_files/refer/cifar10', batch_size=128)
    dm.setup('fit')

    print(dm.train_dataloader())