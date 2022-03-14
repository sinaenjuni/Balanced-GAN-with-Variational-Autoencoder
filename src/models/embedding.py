import torch
import torch.nn as nn



class Embedding_labeled_latent(nn.Module):
    def __init__(self, num_class, latent_dim):
        super(Embedding_labeled_latent, self).__init__()

        self.embeding = nn.Embedding(num_class, latent_dim)

    def forward(self, label, latent):
        embedding = self.embeding(label)
        print(embedding.size())
        embedding = torch.flatten(embedding, 1)
        return torch.multiply(latent, embedding)

if __name__ == "__main__":

    EL = Embedding_labeled_latent(num_class=10, latent_dim=128)

    label = torch.tensor([[1],[2],[3]])
    latent = torch.randn((3, 128))

    print(label.size())
    print(latent.size())

    output = EL(label, latent)
    print(output.size())

