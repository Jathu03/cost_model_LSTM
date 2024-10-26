import copy
import math
import os
import random
import time
import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

def mape_criterion(inputs, targets):
    """Calculate mean absolute percentage error with epsilon smoothing."""
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))

def prepare_batch_for_gnn(batch, device):
    """Convert batch data to GNN format."""
    graphs = []
    for input_data in batch[0]:
        # Extract graph data from input
        node_features, edge_index = input_data
        
        # Create PyG Data object
        graph = Data(
            x=node_features.to(device),
            edge_index=edge_index.to(device),
        )
        graphs.append(graph)
    
    # Combine into a batch
    batched_graphs = Batch.from_data_list(graphs)
    labels = batch[1].to(device)
    
    return batched_graphs, labels

def train_model(
    config,
    model,
    criterion,
    optimizer,
    max_lr,
    dataloader,
    num_epochs,
    log_every,
    logger,
    train_device,
    validation_device,
    max_batch_size=1024,
):
    since = time.time()
    losses = []
    train_loss = 0
    best_loss = math.inf
    best_model = None
    hash = random.getrandbits(16)
    dataloader_size = {"train": 0, "val": 0}
    
    # Calculate dataset sizes
    for item in dataloader["train"]:
        label = item[1]
        dataloader_size["train"] += label.shape[0]
    for item in dataloader["val"]:
        label = item[1]
        dataloader_size["val"] += label.shape[0]
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(dataloader["train"]),
        epochs=num_epochs,
    )
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                device = train_device
            else:
                model.eval()
                device = validation_device if validation_device != "cpu" else validation_device
            
            # Move model to device
            model = model.to(device)
            model.device = device
            
            running_loss = 0.0
            running_attention = 0.0  # Track attention metrics
            pbar = tqdm(dataloader[phase])
            
            for batch_idx, (inputs, labels) in enumerate(pbar):
                # Prepare batch for GNN processing
                graph_batch, labels = prepare_batch_for_gnn((inputs, labels), device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    outputs, attention_weights = model(graph_batch)
                    assert outputs.shape == labels.shape
                    
                    # Calculate loss
                    loss = criterion(outputs, labels) * labels.shape[0] / max_batch_size
                    
                    # Track attention metrics
                    if attention_weights is not None:
                        running_attention += attention_weights.mean().item()
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                # Update progress bar
                pbar.set_description(
                    f"Loss: {loss.item():.3f}, Attention: {running_attention/(batch_idx+1):.3f}"
                )
                
                running_loss += loss.item() * max_batch_size
                
            epoch_loss = running_loss / dataloader_size[phase]
            epoch_attention = running_attention / len(dataloader[phase])
            
            if phase == "val":
                losses.append((train_loss, epoch_loss))
                
                # Save best model
                if epoch_loss <= best_loss:
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    saved_model_path = os.path.join(config.experiment.base_path, "weights/")
                    os.makedirs(saved_model_path, exist_ok=True)
                    
                    full_path = os.path.join(
                        saved_model_path,
                        f"best_model_{config.experiment.name}_{hash:4x}.pt",
                    )
                    logger.debug(f"Saving checkpoint to {full_path}")
                    torch.save(model.state_dict(), full_path)
                
                # Log metrics
                if config.wandb.use_wandb:
                    wandb.log({
                        "best_msle": best_loss,
                        "train_msle": train_loss,
                        "val_msle": epoch_loss,
                        "attention_score": epoch_attention,
                        "epoch": epoch,
                    })
                
                # Print progress
                print(
                    f"Epoch {epoch + 1}/{num_epochs}:  "
                    f"train Loss: {train_loss:.4f}   "
                    f"val Loss: {epoch_loss:.4f}   "
                    f"attention: {epoch_attention:.4f}   "
                    f"time: {time.time() - epoch_start:.2f}s   "
                    f"best: {best_loss:.4f}"
                )
                
                if epoch % log_every == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}:  "
                        f"train Loss: {train_loss:.4f}   "
                        f"val Loss: {epoch_loss:.4f}   "
                        f"attention: {epoch_attention:.4f}   "
                        f"time: {time.time() - epoch_start:.2f}s   "
                        f"best: {best_loss:.4f}"
                    )
            else:
                train_loss = epoch_loss
                scheduler.step()

    time_elapsed = time.time() - since
    
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s   "
        f"best validation loss: {best_loss:.4f}"
    )
    logger.info(
        f"-----> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s   "
        f"best validation loss: {best_loss:.4f}\n"
    )
    
    return losses, best_model