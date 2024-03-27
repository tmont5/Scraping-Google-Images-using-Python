import torch
from torchvision.models import ResNet18_Weights, resnet18
from torch import nn
import torch.nn.functional as F
from typing import Optional


class ResNet18Embedder(nn.Module):
    """Wrapper class for fine-tuning a pretrained ResNet18 to produce embeddings.
    
    Attributes:
        embedding_dim (int): Specifies the output dimension of the embedding.
        freeze_first_two_layers (bool): Indicates whether/not to partially freeze the ResNet18. Good for
                                        regularization when working with small datasets (i.e. not ImageNet).
        add_dropout (bool): Indicates whether/not to add a dropout layer after layer 3.
        checkpoint_path (Optional(str)): If provided, specifies a filepath from which to load a pretrained
                                         ResNet18Embedder. Assumes checkpoint dictionary has a 'state_dict' key.
    """

    def __init__(self, embedding_dim: int = 512, freeze_first_two_layers: bool = True, add_dropout: bool = True, checkpoint_path: Optional[str] = None):
        super(ResNet18Embedder, self).__init__()

        if checkpoint_path is None:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)

            # Edit last linear layer to spit out embedding vectors.
            self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

            if freeze_first_two_layers:
                for p in self.model.layer1.parameters():
                    p.requires_grad = False
                for p in self.model.layer2.parameters():
                    p.requires_grad = False

            if add_dropout:
                dropout_layer = lambda m, inp, out: F.dropout(out, p=0.4, training=m.training)
                self.model.layer3.register_forward_hook(dropout_layer)

        else:
            checkpoint = torch.load(checkpoint_path)
            self.model = resnet18()
            self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        return F.normalize(self.model(x), p=2, dim=1)
