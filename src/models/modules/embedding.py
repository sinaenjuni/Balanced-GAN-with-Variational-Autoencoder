import torch
import torch.nn as nn



class Label_embedding_model(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(Label_embedding_model, self).__init__()

        self.embedding = nn.Embedding(num_classes, latent_dim)

    def forward(self, latent, label):
        embedding = self.embedding(label)
        embedding = torch.flatten(embedding, 1)
        return torch.multiply(latent, embedding)

if __name__ == "__main__":

    EL = Label_embedding_model(num_classes=10, latent_dim=128)

    label = torch.tensor([0, 1, 2, 4])
    latent = torch.randn((4, 128))

    print(label.size())
    print(latent.size())

    output = EL(latent, label)
    print(output.size())

