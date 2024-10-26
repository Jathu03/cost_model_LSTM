from unicodedata import bidirectional
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, global_add_pool
from data_utils import MAX_NUM_TRANSFORMATIONS, MAX_TAGS

def initialization_function_sparse(x):
    return nn.init.sparse_(x, sparsity=0.1)

def initialization_function_xavier(x):
    return nn.init.xavier_uniform_(x)

class GNNCostModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=256,
        num_gnn_layers=4,
        num_attention_heads=8,
        dropout=0.2,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Expression embedding
        self.expr_embedding = nn.Sequential(
            nn.Linear(11, hidden_size),  # 11 is the expression vector size
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformation vector embedding
        self.transform_embedding = nn.Sequential(
            nn.Linear(MAX_TAGS, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_size,
                out_channels=hidden_size // num_attention_heads,
                heads=num_attention_heads,
                dropout=dropout,
                concat=True
            ) for _ in range(num_gnn_layers)
        ])
        
        # Loop level embedding
        self.loop_embedding = nn.Sequential(
            nn.Linear(8, hidden_size),  # 8 is loops_tensor_size
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Computation embedding layers
        self.comp_embedding = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            initialization_function_xavier(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def build_graph(self, tree, comps_embeddings, loops_tensor):
        """Recursively build graph structure from tree."""
        node_features = []
        edge_index = []
        edge_attr = []
        
        def process_node(node, parent_idx=-1):
            current_idx = len(node_features)
            
            # Add loop features
            if "loop_index" in node:
                loop_feat = self.loop_embedding(
                    torch.index_select(loops_tensor, 1, node["loop_index"].to(self.device))
                )
                node_features.append(loop_feat)
                
                # Connect to parent if exists
                if parent_idx >= 0:
                    edge_index.append([parent_idx, current_idx])
                    edge_index.append([current_idx, parent_idx])
            
            # Process computations in this node
            if node["has_comps"]:
                comp_indices = node["computations_indices"].to(self.device)
                selected_comps = torch.index_select(comps_embeddings, 1, comp_indices)
                
                for i in range(selected_comps.size(1)):
                    comp_idx = len(node_features)
                    node_features.append(selected_comps[:, i])
                    
                    # Connect computation to current loop
                    edge_index.append([current_idx, comp_idx])
                    edge_index.append([comp_idx, current_idx])
            
            # Process children recursively
            for child in node["child_list"]:
                process_node(child, current_idx)
        
        # Process all roots
        for root in tree["roots"]:
            process_node(root)
            
        return (
            torch.stack(node_features),
            torch.tensor(edge_index, dtype=torch.long).t().to(self.device)
        )
        
    def forward(self, tree_tensors):
        tree, comps_first, comps_vectors, comps_third, loops_tensor, expr_tree = tree_tensors
        batch_size = comps_first.size(0)
        
        # Embed expressions
        expr_embedding = self.expr_embedding(
            expr_tree.view(batch_size * -1, 11)
        ).view(batch_size, -1, self.hidden_size)
        
        # Embed transformation vectors
        trans_embedding = self.transform_embedding(comps_vectors)
        
        # Combine computation features
        comp_features = torch.cat([
            comps_first.view(batch_size, -1),
            trans_embedding.view(batch_size, -1),
            comps_third.view(batch_size, -1)
        ], dim=1)
        comp_embedding = self.comp_embedding(comp_features)
        
        # Build graph structure
        node_features, edge_index = self.build_graph(
            tree, comp_embedding, loops_tensor
        )
        
        # Apply GAT layers
        x = node_features
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling to get graph-level representations
        graph_repr = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(self.device))
        
        # Combine with computation and expression features
        combined_features = torch.cat([
            graph_repr,
            comp_embedding.mean(dim=1),
            expr_embedding.mean(dim=1)
        ], dim=1)
        
        # Final prediction
        out = self.prediction_head(combined_features)
        return F.leaky_relu(out.squeeze(-1), negative_slope=0.01)
    
    def get_attention_weights(self):
        """Return attention weights from GAT layers for visualization."""
        attention_weights = []
        for layer in self.gat_layers:
            if hasattr(layer, '_alpha'):
                attention_weights.append(layer._alpha)
        return attention_weights