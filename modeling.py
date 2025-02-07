from unicodedata import bidirectional
import torch
from torch import nn
from data_utils import MAX_NUM_TRANSFORMATIONS, MAX_TAGS

def initialization_function_sparse(x):
    return nn.init.sparse_(x, sparsity=0.1)

def initialization_function_xavier(x):
    return nn.init.xavier_uniform_(x)

# Define the architecture of the cost model
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        embedding_size = comp_embed_layer_sizes[-1]
        
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [
            embedding_size * 2 + loops_tensor_size
        ] + comp_embed_layer_sizes[-2:]
        
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        # Create the transformation encoding layers
        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True,
        )
        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        # Create the final regression layers
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(
                    regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
            
        # Create the feed forward netwrok responsible for embedding loop levels
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            initialization_function_xavier(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        # Output layer
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        
        
        initialization_function_xavier(self.predict.weight)
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        # Initialize a tensor to represent the absence of computations at a level in the program tree
        self.no_comps_tensor = nn.Parameter(
            initialization_function_xavier(torch.zeros(1, embedding_size))
        )
        # Initialize a tensor to represent the absence of child loops at a level in the program tree
        self.no_nodes_tensor = nn.Parameter(
            initialization_function_xavier(torch.zeros(1, embedding_size))
        )
        # LSTM to encode computations
        self.comps_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode child loop levels
        self.nodes_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode program roots
        self.roots_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode computations
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # LSTM to encode computation expressions
        self.exprs_embed = nn.LSTM(
            11,
            expr_embed_size,
            batch_first=True,
        )
    # Recursive function to embed a root of the program
    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            # Recusrive call to embed all the children of the loop first if they exist
            nodes_list.append(self.get_hidden_state(
                n, comps_embeddings, loops_tensor))
        
        if nodes_list != []:
            # Pass the embedding of all the child loops through the nodes LSTM
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        
        else: # If there are no child loops contained within this level
            # The nodes embedding is a random vector (no_nodes_tensor) that represents that there are no nodes underneath this level
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )
        if node["has_comps"]:
            # If there are computations contained in this loop, pass them through the computations LSTM
            selected_comps_tensor = torch.index_select(
                comps_embeddings, 
                1, 
                node["computations_indices"].to(self.device)
            )
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(
                selected_comps_tensor
            )
            comps_h_n = comps_h_n.permute(1, 0, 2)
        else: # If there are no child computations contained within this level
            # The computations embedding is a random vector (no_comps_tensor) that represents that there are no computations underneath this level
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                comps_embeddings.shape[0], 
                -1, 
                -1
            )
        # Get the loop vector for this level
        selected_loop_tensor = torch.index_select(
            loops_tensor, 
            1, 
            node["loop_index"].to(self.device)
        )
        # Concatinate the loop vector, computations embedding and nodes (child loops) embedding
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        # Pass the concatinated vector through a feed forward neural network
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x

    def forward(self, tree_tensors):
        # Separate the input tensor
        tree, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree = tree_tensors
        
        # Embed all the expressions in the tree 
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        
        # Expressions embedding layer
        x = functions_comps_expr_tree.view(batch_size* num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # Embed all the computations in the tree
        batch_size, num_comps, __dict__ = comps_tensor_first_part.shape
        
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        vectors = comps_tensor_vectors.to(self.device) # No need to reshape this tensor since we transformed it when loading the data
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)
        
        # Pass the transformation vectors through the vector encoding LSTM
        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # Concatinate the leftover parts from the computatuion, the vectors embedding, and the expression embedding
        x = torch.cat(
            (
                first_part,
                prog_embedding,
                third_part,
                expr_embedding,
            ),
            dim=1,
        ).view(batch_size, num_comps, -1)
        
        # Pass the concatinated vector through a feed forward neural network to extract the final computation embedding vector
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        
        # For each root in the program tree structure 
        roots_list = []
        for root in tree["roots"]:
            # We call the recusrive embedding function to extract the root's embedding
            roots_list.append(
                self.get_hidden_state(
                    root, 
                    comps_embeddings, 
                    loops_tensor
                )
            )
        
        # We concatinate the roots and pass them through an LSTM
        roots_tensor = torch.cat(roots_list, 1)
        lstm_out, (roots_h_n, roots_c_n) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        # Finally, we pass the output of the roots LSTM to the regression layers
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        # We know the speedups to be predicted need to be greater or  equal to zero
        # We use the LeakyRelu to assure this while avoiding the dying ReLu problem
        return self.LeakyReLU(out[:, 0, 0])


'''
import torch
from torch import nn
from data_utils import MAX_NUM_TRANSFORMATIONS, MAX_TAGS

# TreeLSTM Cell Implementation
class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear layers for input and forget gates
        self.W_i = nn.Linear(input_size, hidden_size, bias=True)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=True)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Linear layers for cell state and output
        self.W_o = nn.Linear(input_size, hidden_size, bias=True)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size, bias=True)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x, child_h, child_c):
        # Sum over all child hidden states
        h_sum = torch.sum(child_h, dim=1)
        
        # Compute input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        
        # Compute forget gate for each child
        f = torch.sigmoid(self.W_f(x).unsqueeze(1) + self.U_f(child_h))
        
        # Compute cell state
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h_sum))
        c = i * c_tilde + torch.sum(f * child_c, dim=1)
        
        # Compute output gate and hidden state
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        h = o * torch.tanh(c)
        
        return h, c


# Define the TreeLSTM Model
class Model_TreeLSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        tree_hidden_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.tree_hidden_size = tree_hidden_size
        embedding_size = comp_embed_layer_sizes[-1]

        # TreeLSTM Cell
        self.tree_lstm = TreeLSTMCell(tree_hidden_size, tree_hidden_size)

        # Embedding layers
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()

        comp_embed_layer_sizes = [
            input_size + expr_embed_size + loops_tensor_size
        ] + comp_embed_layer_sizes

        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        regression_layer_sizes = [comp_embed_layer_sizes[-1]] + [output_size]
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(
                    regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True
                )
            )
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[-1]))

        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)
        self.predict = nn.Linear(tree_hidden_size, output_size, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)

    def forward(self, tree_tensors):
        # Unpack tree tensors
        tree, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree = tree_tensors

        # Expressions embedding
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )

        # Compute embeddings for computations
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)
        x = torch.cat((first_part, third_part, expr_embedding), dim=1)

        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x.view(batch_size, num_comps, -1)

        # Compute embeddings for tree nodes using TreeLSTM
        def compute_node_embedding(node):
            if "child_list" in node and len(node["child_list"]) > 0:
                child_embeddings = torch.stack(
                    [compute_node_embedding(child) for child in node["child_list"]], dim=1
                )
                child_h, child_c = torch.chunk(child_embeddings, 2, dim=2)
            else:
                child_h = torch.zeros(
                    batch_size, 1, self.tree_hidden_size, device=self.device
                )
                child_c = torch.zeros(
                    batch_size, 1, self.tree_hidden_size, device=self.device
                )
            node_h, node_c = self.tree_lstm(
                comps_embeddings[:, node["index"]],
                child_h,
                child_c,
            )
            return torch.cat((node_h, node_c), dim=2)

        root_embeddings = torch.stack(
            [compute_node_embedding(root) for root in tree["roots"]], dim=1
        )

        # Final regression layer
        x = root_embeddings.sum(dim=1)  # Aggregate root embeddings
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        return self.LeakyReLU(self.predict(x).squeeze())
    '''
