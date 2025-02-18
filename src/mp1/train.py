import torch
import tqdm
import os
import numpy as np
import wandb
from torch.utils.data import DataLoader
from datasets.lane_dataset import LaneDataset
from models.enet import ENet
from models.losses import compute_loss
from utils.visualization import visualize_first_prediction
from torch.optim import Adam

# Configurations
BATCH_SIZE = 
LR = 
EPOCHS = 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH =  "/opt/data/TUSimple"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def validate(model, val_loader):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    val_loss = 0
    binary_losses = []
    instance_losses = []

    with torch.no_grad():
        for images, binary_labels, instance_labels in tqdm.tqdm(val_loader, desc="Validating"):
            images = images.to(DEVICE)
            binary_labels = binary_labels.to(DEVICE)
            instance_labels = instance_labels.to(DEVICE)

            binary_logits, instance_embeddings = model(images)

            binary_loss, instance_loss = compute_loss(
                binary_output=binary_logits,
                instance_output=instance_embeddings,
                binary_label=binary_labels,
                instance_label=instance_labels,
            )
            loss = binary_loss + instance_loss

            val_loss += loss.item()
            binary_losses.append(binary_loss.item())
            instance_losses.append(instance_loss.item())

    mean_binary_loss = np.mean(binary_losses)
    mean_instance_loss = np.mean(instance_losses)
    total_loss = val_loss / len(val_loader)

    return mean_binary_loss, mean_instance_loss, total_loss

def train():
    """
    Train the ENet model on the training dataset and log results to Weights & Biases.
    """
    wandb.init(
        project="lane-detection",
        name="ENet-Training",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "model": "ENet"
        }
    )

    # TODO: Data preparation: Load and preprocess the training and validation datasets.
    # Hint: Use the LaneDataset class and PyTorch's DataLoader.
    ################################################################################
    # train_dataset = ...
    # train_loader = DataLoader(...)

    # val_dataset = ...
    # val_loader = DataLoader(...)
    ################################################################################

    # Model and optimizer initialization
    enet_model = ENet(binary_seg=2, embedding_dim=4).to(DEVICE)
    
    
    # TODO: Initialize the Adam optimizer with appropriate learning rate and weight decay.
    ################################################################################
    # optimizer = ...
    
    ################################################################################


    def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
        """
        Save model checkpoints during training.
        """
        checkpoint_path = os.path.join(checkpoint_dir, f"enet_checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        wandb.save(checkpoint_path)  # Save to W&B
        print(f"Checkpoint saved at {checkpoint_path}")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        enet_model.train()
        epoch_loss = 0
        binary_losses = []
        instance_losses = []

        for batch_idx, (images, binary_labels, instance_labels) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            
            # TODO: Complete the training step for a single batch.
            ################################################################################
            # Hint:
            # 1. Move `images`, `binary_labels`, and `instance_labels` to the correct device (e.g., GPU).
            # 2. Perform a forward pass using `enet_model` to get predictions (`binary_logits` and `instance_embeddings`).
            # 3. Compute the binary and instance losses using `compute_loss`.
            # 4. Sum the losses (`loss = binary_loss + instance_loss`) for backpropagation.
            # 5. Zero out the optimizer gradients, backpropagate the loss, and take an optimizer step.





            ################################################################################
            
            
            # Log visualizations for the first batch of the epoch
            if batch_idx == 0:
                combined_row = visualize_first_prediction(
                    images.cpu(),
                    binary_logits.cpu(),
                    instance_embeddings.cpu(),
                    binary_labels.cpu(),
                    instance_labels.cpu()
                )
                wandb.log({"visualization": wandb.Image(combined_row, caption=f"Epoch {epoch} - Batch {batch_idx}")})

        # Epoch-wise logging
        mean_binary_loss = np.mean(binary_losses)
        mean_instance_loss = np.mean(instance_losses)
        total_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch}/{EPOCHS}: "
              f"Binary Loss = {mean_binary_loss:.4f}, "
              f"Instance Loss = {mean_instance_loss:.4f}, "
              f"Total Loss = {total_loss:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_binary_loss": mean_binary_loss,
            "train_instance_loss": mean_instance_loss,
            "train_total_loss": total_loss
        })

        # Validation after each epoch
        # TODO: Perform validation after each epoch
        # Hint:
        # Call the `validate` function, passing the model and validation data loader.
        ################################################################################
        # val_binary_loss, val_instance_loss, val_total_loss = ...
        ################################################################################
        print(f"Validation Results - Epoch {epoch}: "
              f"Binary Loss = {val_binary_loss:.4f}, "
              f"Instance Loss = {val_instance_loss:.4f}, "
              f"Total Loss = {val_total_loss:.4f}")

        wandb.log({
            "val_binary_loss": val_binary_loss,
            "val_instance_loss": val_instance_loss,
            "val_total_loss": val_total_loss
        })

        save_checkpoint(enet_model, optimizer, epoch, CHECKPOINT_DIR)

    wandb.finish()

if __name__ == '__main__':
    train()
