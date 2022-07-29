import torch
from torch.optim import Adam
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb
from pytorch_lightning.loggers import WandbLogger

from sr.model_resnet import Generator, Discriminator
from sr.dataset import DataModule_
from sr.misc import MODULES
import sr.losses as losses

class MyModel(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 g_conv_dim,
                 d_conv_dim,
                 d_embed_dim,
                 img_channels,
                 num_classes,
                 module,
                 learning_rate,
                 **kargs
                 ):
        super(MyModel, self).__init__()
        self.save_hyperparameters()

        self.G = Generator(z_dim=self.hparams.latent_dim, img_size=64, g_conv_dim=self.hparams.g_conv_dim, num_classes=self.hparams.num_classes, g_init='ortho', MODULES=module)
        self.D = Discriminator(img_size=64, d_conv_dim=self.hparams.d_conv_dim, d_embed_dim=self.hparams.d_embed_dim, num_classes=self.hparams.num_classes, d_init='ortho', MODULES=module)

        self.fid = FrechetInceptionDistance()
        # self.ins = InceptionScore()
        # self.metric_loss = losses.ContraGAN_loss()

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
            # d_loss = self.d_hinge(r_logit, f_logit)
            d_loss = losses.d_bce_loss(r_logit, f_logit)
            # dm_loss = self.metric_loss(r_feature, real_embed, labels)
            self.log('d_loss', d_loss, prog_bar=True, logger=True, on_epoch=True)
            # self.log('dm_loss', dm_loss, prog_bar=True, logger=True, on_epoch=True)
            return d_loss

        if optimizer_idx == 1:
            g_logit, g_feature, g_embed = self.D(self(z, fake_labels), fake_labels)
            # g_loss = self.g_hinge(g_logit)
            g_loss = losses.g_bce_loss(g_logit)
            # gm_loss = self.metric_loss(g_feature, g_embed)
            self.log('g_loss', g_loss, prog_bar=True, logger=True, on_epoch=True)
            # self.log('gm_loss', gm_loss, prog_bar=True, logger=True, on_epoch=True)
            return g_loss


    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = (((imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        self.fid.update(imgs, real=True)


    def validation_epoch_end(self, outputs):
        for c_ in range(self.hparams.num_classes):
            for i in range(10):
                z = torch.randn((100, self.hparams.latent_dim)).to(self.device)
                labels = torch.ones((100, ), dtype=torch.long, device=self.device) * c_
                gend_imgs = self(z, labels)
                gend_imgs = (((gend_imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
                self.fid.update(gend_imgs, real=False)

        self.log('fid', self.fid.compute(), logger=True, prog_bar=True, on_epoch=True)
        self.fid.reset()

        z = torch.randn((100, self.hparams.latent_dim)).to(self.device)
        label = torch.arange(0, 10, dtype=torch.long, device=self.device).repeat(10)
        gened_imgs = self(z, label)
        self.logger.log_image("img", [gened_imgs], self.trainer.current_epoch)


    def configure_optimizers(self):
        # discriminator training first
        optimizer_d = Adam(self.D.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.999))
        optimizer_g = Adam(self.G.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.999))

        # return [optimizer_d, optimizer_g], []
        return [{'optimizer': optimizer_d, 'frequency': 5},
                {'optimizer': optimizer_g, 'frequency': 1}]


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--latent_dim', default=80, type=int)
    parser.add_argument('--g_conv_dim', default=32, type=int)
    parser.add_argument('--d_conv_dim', default=32, type=int)
    parser.add_argument('--d_embed_dim', default=256, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--img_channels', default=3, type=int)
    parser.add_argument('--path_train', default='/home/dblab/sin/save_files/refer/ebgan_cifar10', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--module', default=MODULES, type=object)

    parser = pl.Trainer.add_argparse_args(parser)
    # parser = MyModel.add_model_specific_args(parser)
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
        **vars(args)
        # latent_dim=args.latent_dim,
        # img_channels=args.img_channels,
        # MODULES=MODULES,
        # dm.input_vocab_size,
        # dm.output_vocab_size,
        # args.n_layers,
        # args.d_model,  # dim. in attemtion mechanism
        # args.n_heads,
        # dm.padding_idx,
        # learning_rate=args.learning_rate,
        # args.n_split
    )

    # ------------
    # training
    # ------------
    wandb.login(key='6afc6fd83ea84bf316238272eb71ef5a18efd445')
    # wandb.init(project='GAN', name='our_adv_bce')
    wandb.init(project='GAN')
    wandb_logger = WandbLogger(project="GAN")

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=500,
        callbacks=[pl.callbacks.ModelCheckpoint(filename="our-{epoch:02d}-{fid}",
                                                monitor="fid", mode='min')],
        logger=wandb_logger,
        # logger=False,
        strategy='ddp',
        accelerator='gpu',
        gpus=[0, 1],
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