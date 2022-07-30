import torch
import torch.nn as nn
import torch.nn.functional as F


class Our_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim_fn = nn.CosineSimilarity(dim=-1)
        self.temp = 1

    def _calculate_similarity_matrix(self, v1, v2):
        return self.sim_fn(v1.unsqueeze(1), v2.unsqueeze(0))

    def _remove_diagonal(self, m):
        batch_size = m.size(0)
        mask_diag = torch.ones_like(m).fill_diagonal_(0).to(torch.bool)
        return m[mask_diag].view(batch_size, -1)

    def _get_mask(self, label):
        return torch.eq(label.view(-1, 1), label.view(-1, 1).T).to(torch.int)

    def forward(self, feature, embed, label):
        mask = self._get_mask(label)
        mask_pos = self._remove_diagonal(mask)

        sim_f2f = self._calculate_similarity_matrix(feature, feature)
        sim_f2f = torch.exp(self._remove_diagonal(sim_f2f) / self.temp )
        sim_masked_f2f = mask_pos * sim_f2f

        sim_e2f = self._calculate_similarity_matrix(embed, feature)
        sim_e2f = torch.exp(sim_e2f / self.temp)
        sim_masked_e2f = mask * sim_e2f

        numerator   = sim_masked_f2f.sum(1) + sim_masked_e2f.sum(1)
        denominator = sim_f2f.sum(1) + sim_e2f.sum(1)

        return -torch.log( numerator / denominator ).mean()




class ContraGAN_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim_fn = nn.CosineSimilarity(dim=-1)
        self.temp = 1

    def _calculate_similarity_matrix(self, v1, v2):
        return self.sim_fn(v1.unsqueeze(1), v2.unsqueeze(0))

    def _remove_diagonal(self, m):
        batch_size = m.size(0)
        mask_diag = torch.ones_like(m).fill_diagonal_(0).to(torch.bool)
        return m[mask_diag].view(batch_size, -1)

    def _get_mask(self, label):
        return torch.eq(label.view(-1, 1), label.view(-1, 1).T).to(torch.int)

    def forward(self, feature, embed, label):
        mask = self._get_mask(label)
        mask_pos = self._remove_diagonal(mask)

        sim_f2f = self._calculate_similarity_matrix(feature, feature)
        sim_f2f = torch.exp(self._remove_diagonal(sim_f2f) / self.temp )
        sim_masked_f2f = mask_pos * sim_f2f

        sim_e2f = self.sim_fn(embed, feature)
        sim_e2f = torch.exp(sim_e2f / self.temp)

        numerator = sim_e2f + sim_masked_f2f.sum(1)
        denominator = torch.cat([sim_e2f.unsqueeze(1), sim_f2f], dim=1).sum(1)
        return -torch.log( numerator / denominator ).mean()


def d_hinge_loss(logit_real, logit_fake):
    return torch.mean(F.relu(1. - logit_real)) + torch.mean(F.relu(1. + logit_fake))

def g_hinge_loss(logit_fake):
    return -torch.mean(logit_fake)


def d_bce_loss(logit_real, logit_fake):
    real_loss = F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
    fake_loss = F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))
    return real_loss + fake_loss

def g_bce_loss(logit_fake):
    fake_loss = F.binary_cross_entropy_with_logits(logit_fake, torch.ones_like(logit_fake))
    return fake_loss