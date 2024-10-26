import os
import io
import logging
import random
import gc
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import torch
from torch_geometric.data import DataLoader
from data_utils import *
from modeling import GNNCostModel
from train_utils import *

def load_graph_batches_from_path(train_paths, train_devices, val_paths):
    """Load and prepare graph batches from saved paths."""
    assert(len(train_paths)>0 and len(val_paths)>0)
    assert(len(train_paths) == len(train_devices))
    train_bl = []
    val_bl = []
    
    # Load training data
    for index, path in enumerate(train_paths):
        if os.path.exists(path):
            print(f"Loading training data from {path} into {train_devices[index]}")
            with open(path, "rb") as file:
                batch_data = torch.load(path, map_location=train_devices[index])
                # Convert to graph format if needed
                graph_batch = convert_to_graph_batch(batch_data, train_devices[index])
                train_bl += graph_batch
                
    # Load validation data
    for path in val_paths:
        if os.path.exists(path):
            print(f"Loading validation data from {path} into CPU")
            with open(path, "rb") as file:
                batch_data = torch.load(path, map_location="cpu")
                # Convert to graph format if needed
                graph_batch = convert_to_graph_batch(batch_data, "cpu")
                val_bl += graph_batch
    
    # Shuffle datasets
    random.shuffle(train_bl)
    random.shuffle(val_bl)
    
    return train_bl, val_bl

def convert_to_graph_batch(batch_data, device):
    """Convert traditional batch data to graph format."""
    graph_batch = []
    for data in batch_data:
        # Extract features and build graph structure
        inputs, labels = data
        tree, comps_first, comps_vectors, comps_third, loops_tensor, expr_tree = inputs
        
        # Create graph data structure
        node_features, edge_index = build_graph_from_tree(
            tree, comps_first, comps_vectors, comps_third, 
            loops_tensor, expr_tree, device
        )
        
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=labels
        )
        graph_batch.append(graph_data)
    
    return graph_batch

@hydra.main(config_path="conf", config_name="config")
def main(conf):
    # Setup logging
    log_filename = [part for part in conf.training.log_file.split('/') if len(part) > 3][-1]
    log_folder_path = os.path.join(conf.experiment.base_path, "logs/")
    os.makedirs(log_folder_path, exist_ok=True)
    
    log_file = os.path.join(log_folder_path, log_filename)
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s: %(message)s'
    )
    logging.info(f"Starting experiment {conf.experiment.name}")
    
    # Set devices
    train_device = torch.device(conf.training.training_gpu)
    validation_device = torch.device(conf.training.validation_gpu)
    
    # Initialize GNN model
    logging.info("Defining the GNN model")
    model = GNNCostModel(
        input_size=conf.model.input_size,
        hidden_size=conf.model.hidden_size,
        num_gnn_layers=conf.model.num_gnn_layers,
        num_attention_heads=conf.model.num_attention_heads,
        dropout=conf.model.dropout,
        device=train_device
    )
    
    # Load pretrained weights if continuing training
    if conf.training.continue_training:
        print(f"Continue training using model from {conf.training.model_weights_path}")
        model.load_state_dict(
            torch.load(conf.training.model_weights_path, map_location=train_device)
        )
    
    # Enable gradient tracking
    for param in model.parameters():
        param.requires_grad = True
    
    # Load training data
    logging.info("Loading datasets")
    train_data = []
    val_data = []
    
    # Load training data parts
    train_cpu_path = os.path.join(
        conf.experiment.base_path, 
        "batched/train/", 
        f"{Path(conf.data_generation.train_dataset_file).parts[-1][:-4]}_CPU.pt"
    )
    train_gpu_path = os.path.join(
        conf.experiment.base_path, 
        "batched/train/", 
        f"{Path(conf.data_generation.train_dataset_file).parts[-1][:-4]}_GPU.pt"
    )
    
    # Load CPU part if exists
    if os.path.exists(train_cpu_path):
        print(f"Loading CPU training data from {train_cpu_path}")
        with open(train_cpu_path, "rb") as file:
            train_cpu_data = torch.load(train_cpu_path, map_location="cpu")
            train_data += convert_to_graph_batch(train_cpu_data, "cpu")
    
    # Load GPU part
    print(f"Loading GPU training data into {conf.training.training_gpu}")
    with open(train_gpu_path, "rb") as file:
        train_gpu_data = torch.load(train_gpu_path, map_location=train_device)
        train_data += convert_to_graph_batch(train_gpu_data, train_device)
    
    # Load validation data parts
    val_cpu_path = os.path.join(
        conf.experiment.base_path, 
        "batched/valid/", 
        f"{Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]}_CPU.pt"
    )
    val_gpu_path = os.path.join(
        conf.experiment.base_path, 
        "batched/valid/", 
        f"{Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]}_GPU.pt"
    )
    
    # Load validation data similarly
    if os.path.exists(val_cpu_path):
        with open(val_cpu_path, "rb") as file:
            val_cpu_data = torch.load(val_cpu_path, map_location="cpu")
            val_data += convert_to_graph_batch(val_cpu_data, "cpu")
            
    with open(val_gpu_path, "rb") as file:
        val_gpu_data = torch.load(val_gpu_path, map_location=validation_device)
        val_data += convert_to_graph_batch(val_gpu_data, validation_device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=conf.data_generation.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=conf.data_generation.batch_size, 
        shuffle=False
    )
    
    # Setup training
    criterion = mape_criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=conf.training.lr, 
        weight_decay=0.15e-1
    )
    logger = logging.getLogger()
    
    # Initialize wandb if enabled
    if conf.wandb.use_wandb:
        wandb.init(name=conf.experiment.name, project=conf.wandb.project)
        wandb.config = dict(conf)
        wandb.watch