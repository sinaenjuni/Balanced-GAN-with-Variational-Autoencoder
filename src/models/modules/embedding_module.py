import torch
import torch.nn as nn



class Embedding_module(nn.Module):
    def __init__(self, num_class, latent_dim):
        super(Embedding_module, self).__init__()
        self.embedding = nn.Sequential(nn.Embedding(num_class, latent_dim),
                                       nn.Flatten(start_dim=1))

    def forward(self, latent, label):
        embedding = self.embedding(label)
        return torch.multiply(latent, embedding)

if __name__ == "__main__":

    EL = Embedding_module(num_class=10, latent_dim=128)

    label = torch.tensor([0, 1, 2, 4])
    latent = torch.randn((4, 128))

    print(label.size())
    print(latent.size())

    output = EL(latent, label)
    print(output.size())

