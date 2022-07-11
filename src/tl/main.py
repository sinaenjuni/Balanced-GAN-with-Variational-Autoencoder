import torch
from torch.optim import Adam
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from model_deep_conv import Generator_, Discriminator_
from torchvision.utils import make_grid
from collections import OrderedDict
from dataset import DataModule_
from misc import MODULES
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


class MyModel(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 img_channels,
                 MODULES,
                 learning_rate
                 ):
        super(MyModel, self).__init__()
        self.save_hyperparameters()

        self.G = Generator_(latent_dim=self.hparams.latent_dim,
                            img_channels=self.hparams.img_channels,
                            MODULES=self.hparams.MODULES)
        self.D = Discriminator_(img_channels=self.hparams.img_channels,
                                MODULES=self.hparams.MODULES)

        # print(self.device)
        self.fid = FrechetInceptionDistance()
        # self.ins = InceptionScore()
        # loss
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()


    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


    def forward(self, z):
        return self.G(z)


    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(imgs)

        if optimizer_idx == 0:
            self.gened_imgs = self(z)

            sample_imgs = self.gened_imgs[:10]
            grid = make_grid(sample_imgs)
            self.logger.experiment.add_image('train_gened_imgs', grid, 0)

            valid = torch.ones(imgs.size(0)).type_as(imgs)

            g_loss = self.adversarial_loss(self.D(self.gened_imgs), valid)

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        if optimizer_idx == 1:
            fake = torch.zeros(imgs.size(0)).type_as(imgs)
            valid = torch.ones(imgs.size(0)).type_as(imgs)

            real_loss = self.adversarial_loss(self.D(imgs), valid)
            fake_loss = self.adversarial_loss(self.D(self(z).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(imgs)
        gend_imgs = self(z)

        imgs      = (((imgs      * 0.5) + 0.5) * 255.).to(torch.uint8)
        gend_imgs = (((gend_imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        # imgs = imgs + 0.5
        # print(imgs.shape, imgs.min(), imgs.max())
        # print(gend_imgs.shape, gend_imgs.min(), gend_imgs.max())


        self.fid.update(imgs, real=True)
        self.fid.update(gend_imgs, real=False)

        # self.ins.update(gend_imgs)

        # return metrics

    # def validation_step_end(self, metrics):
        # print(self.device)

        # print(metrics["val_loss"].mean())
        # print(self.ins.compute())
        pass

    def validation_epoch_end(self, val_step_outputs):
        self.log('valid_fid_epoch',self.fid.compute())
        self.fid.reset()

    #     val_logit = torch.cat([x['val_logit'] for x in val_step_outputs])
    #     val_label = torch.cat([x['val_label'] for x in val_step_outputs])
    #
    #     loss = self.criterion(val_logit, val_label)
    #     acc = FM.accuracy(val_logit, val_label.long(), threshold=0.5)
    #     auc = FM.auroc(val_logit, val_label.long(), pos_label=1)
    #
    #     log_dict = {"val_loss": loss,
    #                 "val_acc": acc,
    #                 "val_auc": auc}
    #     self.log_dict(log_dict, on_epoch=True, prog_bar=True, logger=True)



    # def test_step(self, batch, batch_idx):
    #     miRNA_vec, Gene_vec, label = batch
    #
    #     logit, _ = self(miRNA_vec, Gene_vec)  # [B, output_vocab_size]
    #     loss = self.criterion(logit, label)
    #
    #     ## get predicted result
    #     # prob = F.softmax(logits, dim=-1)
    #     acc = FM.accuracy(logit, label.long(), threshold=0.5)
    #     auc = FM.auroc(logit, label.long(), pos_label=1)
    #
    #     metrics = {'test_acc': acc, 'test_loss': loss, 'test_auc': auc}
    #     self.log_dict(metrics, on_epoch=True)
    #     return metrics

    def configure_optimizers(self):
        optimizer_g = Adam(self.G.parameters(), lr=self.hparams.learning_rate)
        optimizer_d = Adam(self.D.parameters(), lr=self.hparams.learning_rate)

        return [optimizer_g, optimizer_d], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ATTENTION")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--img_channels', default=3, type=int)
    parser.add_argument('--path_train', default='/home/dblab/sin/save_files/refer/ebgan_cifar10', type=str)
    # parser.add_argument('--d_model', default=32, type=int)  # dim. for attention model
    # parser.add_argument('--n_heads', default=4, type=int)  # number of multi-heads
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
        learning_rate=args.learning_rate
        # args.n_split
    )

    # ------------
    # training
    # ------------
    # trainer = pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(
        max_epochs=100,
        # callbacks=[EarlyStopping(monitor='val_loss')],
        gpus=-1  # if you have gpu -- set number, otherwise zero
    )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    # result = trainer.test(model, datamodule=dm)
    # print(result)



if __name__ == '__main__':
    cli_main()