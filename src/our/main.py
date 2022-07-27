import torch
from torch.optim import Adam
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from model_resnet import Generator, Discriminator
from torchvision.utils import make_grid
from collections import OrderedDict
from dataset import DataModule_
from misc import MODULES
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import wandb
from pytorch_lightning.loggers import WandbLogger


class MyModel(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 img_channels,
                 MODULES,
                 learning_rate,
                 ):
        super(MyModel, self).__init__()
        self.save_hyperparameters()

        self.G = Generator(z_dim=self.hparams.latent_dim, img_size=64, g_conv_dim=48, num_classes=10, g_init='ortho', MODULES=MODULES)

        self.D = Discriminator(img_size=64, d_conv_dim=48, d_embed_dim=512, num_classes=10, d_init='ortho', MODULES=MODULES)

        # print(self.device)
        self.fid = FrechetInceptionDistance()
        # self.ins = InceptionScore()
        # loss
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()


    # def d_loss(self, real_logits, fake_logits):
    #     real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits).to(self.device))
    #     fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits).to(self.device))
    #     return real_loss + fake_loss

    def d_hinge(self, logit_real, logit_fake):
        return torch.mean(F.relu(1. - logit_real)) + torch.mean(F.relu(1. + logit_fake))

    def g_hinge(self, logit_fake):
        return -torch.mean(logit_fake)

    def forward(self, z, labels):
        return self.G(z, labels)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim).to(self.device)
        fake_labels = torch.randint(0, 10, size=(batch_size,), device=self.device)

        if optimizer_idx == 0:
            r_logit, r_feature, real_embed = self.D(imgs, labels)
            f_logit, f_feature, fake_embed = self.D(self(z, labels).detach(), labels)
            d_loss = self.d_hinge(r_logit, f_logit)
            self.log('d_loss', d_loss, prog_bar=True, logger=True, on_epoch=True)
            return d_loss

        if optimizer_idx == 1:
            g_logit, g_feature, g_embed = self.D(self(z, fake_labels), fake_labels)
            g_loss = self.g_hinge(g_logit)
            self.log('g_loss', g_loss, prog_bar=True, logger=True, on_epoch=True)
            return g_loss


    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)

        z = torch.randn((batch_size, self.hparams.latent_dim)).to(self.device)
        labels = torch.randint(0, 10, size=(batch_size,), device=self.device)
        # label = torch.arange(0, 9, dtype=torch.long).repeat(100).to(self.device)
        gend_imgs = self(z, labels)

        imgs = (((imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        gend_imgs = (((gend_imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        self.fid.update(imgs, real=True)
        self.fid.update(gend_imgs, real=False)


    def validation_epoch_end(self, outputs):
        # print('valid_fid_epoch', self.fid.compute())
        self.log('fid', self.fid.compute(), logger=True, prog_bar=True, on_epoch=True)
        self.fid.reset()

        # z = torch.randn((100, self.hparams.latent_dim)).to(self.device)
        # label = torch.arange(0, 10, dtype=torch.long).repeat(10).to(self.device)
        # gened_imgs = self(z, label)
        # self.logger.log_image("img", [gened_imgs], self.trainer.current_epoch)


    def configure_optimizers(self):
        # discriminator training first
        optimizer_d = Adam(self.D.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.999))
        optimizer_g = Adam(self.G.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.999))

        return [optimizer_d, optimizer_g], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ATTENTION")
        parser.add_argument('--learning_rate', type=float, default=0.0002)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--latent_dim', default=80, type=int)
    parser.add_argument('--img_channels', default=3, type=int)
    parser.add_argument('--path_train', default='/home/dblab/sin/save_files/refer/ebgan_cifar10', type=str)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    # parser.add_argument('--d_model', default=32, type=int)  # dim. for attention model
    # parser.add_argument('--n_heads', default=4, type=int)  # number of multi-sheads
    # parser.add_argument('--n_split', default=8, type=int)  # number of data seq
    # parser.add_argument('--n_layers', default=8, type=int)  # number of encoder layer

    parser = pl.Trainer.add_argparse_args(parser)
    parser = MyModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = DataModule_.from_argparse_args(args)
    # dm = GeneRNADataModule.from_argparse_args(args)
    # iter(dm.train_dataloader()).next()  # <for testing

    # ------------
    # model
    # ------------
    model = MyModel(
        latent_dim=args.latent_dim,
        img_channels=args.img_channels,
        MODULES=MODULES,
        # dm.input_vocab_size,
        # dm.output_vocab_size,
        # args.n_layers,
        # args.d_model,  # dim. in attemtion mechanism
        # args.n_heads,
        # dm.padding_idx,
        learning_rate=args.learning_rate,
        # args.n_split
    )

    # ------------
    # training
    # ------------
    # wandb.login(key='6afc6fd83ea84bf316238272eb71ef5a18efd445')
    # wandb.init(project='GAN', name='our_adv')
    # wandb_logger = WandbLogger(project="GAN")

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=500,
        callbacks=[pl.callbacks.ModelCheckpoint(filename="our-{epoch:02d}-{fid}",
                                                monitor="fid", mode='min')],
        # logger=wandb_logger,
        logger=False,
        strategy='ddp',
        accelerator='gpu',
        gpus=[1,2],
        check_val_every_n_epoch=10
    )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    # result = trainer.test(model, datamodule=dm)
    # print(result)



if __name__ == '__main__':
    cli_main()