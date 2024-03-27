import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict
import torch
from tqdm import tqdm
from torch.nn import TripletMarginLoss
from utils.triplet_generator import pick_index_of_positive_sample, pick_index_of_negative_sample


def train_embedding_network_with_triplet_loss(
        model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        num_epochs: int = 20, lr: float = 1e-3, momentum: float = 0.9,
        margin: float = 1., validate_every: int = 100, checkpoint: bool = False,
        save_path: str = 'embedding_model.pt', best_achieved_loss: float = float('inf'),
    ) -> Dict[str, List]:
    """Train an embedding model using triplet loss.
    
    Args:
        model (nn.Module): An embedding neural network to train.
        train_loader (DataLoader): Dataloader for the training data.
        val_loader (DataLoader): Dataloader for the validation data.
        num_epochs (int): Number of passes over the data to make while training.
        lr (float): Learning rate parameter for SGD.
        momentum (float): Momentum parameter for SGD.
        margin (float): Margin parameter for Triplet Loss.
        validate_every (int): Specifies number of training batches between subsequent validation loops.
        checkpoint (bool): Indicates whether/not to save model state dict while training.
        save_path (str): Filepath at which to save model. Should only be specified when checkpoint is True.
        best_achieved_loss (float): Loss value to compare against when determining whether/not to save the current model weights.

    Returns:
        training_history (dict): Dictionary with training history of model. 
    """
    if checkpoint and save_path is None:
        raise ValueError("Must specify `save_path` if `checkpoint` is set to True.")
    if not checkpoint and save_path is not None:
        raise ValueError("Save path should not be specified if `checkpoint` is set to False.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    triplet_loss = TripletMarginLoss(margin)
    params_to_train = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params_to_train, lr, momentum)

    train_losses = []
    train_losses_epoch = []
    val_losses = []

    progress_bar = tqdm(total=num_epochs * len(train_loader), position=0, leave=False)

    for epoch in range(num_epochs):
        
        model.train()
        epoch_loss = 0

        for batch, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(device), labels.numpy()
            outputs = model(inputs)
        
            positive_index_picker = lambda i: pick_index_of_positive_sample(i, labels)
            negative_index_picker = lambda i: pick_index_of_negative_sample(i, labels)
            anchor_indices = torch.arange(len(inputs))
            positive_indices = list(map(positive_index_picker, anchor_indices))
            negative_indices = list(map(negative_index_picker, anchor_indices))
            loss = triplet_loss(outputs[anchor_indices], outputs[positive_indices], outputs[negative_indices])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item() / train_loader.batch_size)
            epoch_loss += loss.item()

            if batch % validate_every == 0:

                model.eval()
                with torch.no_grad():

                    val_loss = 0
                    for x, y in val_loader:
                        x, y = x.to(device), y.numpy()
                        y_hat = model(x).squeeze(-1).squeeze(-1)

                        positive_index_picker = lambda i: pick_index_of_positive_sample(i, y)
                        negative_index_picker = lambda i: pick_index_of_negative_sample(i, y)
                        anchor_indices = torch.arange(len(x))
                        positive_indices = list(map(positive_index_picker, anchor_indices))
                        negative_indices = list(map(negative_index_picker, anchor_indices))

                        val_loss += triplet_loss(y_hat[anchor_indices], y_hat[positive_indices], y_hat[negative_indices]).item()

                    val_losses.append(val_loss / (len(val_loader) * val_loader.batch_size))

                progress_bar.set_description(
                    f"Epoch: {epoch + 1}, Batch: {batch}, Train Loss (last {validate_every} batches): {train_losses[-1]:.4f}, Val. Loss: {val_losses[-1]:.4f}"
                )
                model.train()
                
            progress_bar.update(1)

        avg_loss_epoch = epoch_loss / (len(train_loader) * train_loader.batch_size)
        train_losses_epoch.append(avg_loss_epoch)

        if checkpoint and avg_loss_epoch < best_achieved_loss:

            best_achieved_loss = avg_loss_epoch

            torch.save({
                'epoch': epoch + 1,
                'best_achieved_avg_loss': best_achieved_loss,
                'state_dict': model.state_dict(),
            }, save_path)

    training_history = {
        'train_losses': train_losses,
        'train_losses_epoch': train_losses_epoch,
        'val_losses': val_losses,
    }
    
    return training_history
