import copy
import json
from msilib import make_id
import pickle
import random
import re
import gc
import sys
import multiprocessing
import shutil
#import resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import enum
import os, psutil
import sympy

# An exception to limit the maximum number of allowed transformations 
class NbTranformationException(Exception):
    pass

# An exception to limit the maximum number of read-write accesses. 
class NbAccessException(Exception):
    pass

# An exception to limit the maximum number of nested loops. Currently set to 5.
class LoopsDepthException(Exception):
    pass

# Maximum sequence of transformations (reversal, interchange and skewing) allowed. Currently set to 4 
MAX_NUM_TRANSFORMATIONS = 4

# Maximum size of the tags vector representing each transformation
MAX_TAGS = 16

# Maximum depth of a loop nest for each computation
MAX_DEPTH = 5

# Maximum length of expressions in the dataset
MAX_EXPR_LEN = 66

# Creates a template for the input representation

def get_graph_representation_template(program_dict, train_device="cpu"):
    # Set the max and min number of accesses allowed 
    max_accesses = 15
    min_accesses = 0

    # Create lists to hold nodes and edges for the graph
    nodes = []
    edges = []

    # Get the program JSON representation
    program_json = program_dict["program_annotation"]
    
    # Get the computations (program statements) dictionary and order them according to the absolute_order attribute
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )

    # For each computation in the program
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        
        # Check if the computation accesses conform to the minimum and maximum allowed
        if len(comp_dict["accesses"]) > max_accesses or len(comp_dict["accesses"]) < min_accesses:
            continue  # Skip this computation if it doesn't conform
        
        # Create a node for the computation
        node = {
            "name": comp_name,
            "is_reduction": comp_dict["comp_is_reduction"],
            "iterators": comp_dict["iterators"],
            "write_buffer_id": comp_dict["write_buffer_id"],
            "index": comp_index
        }
        nodes.append(node)

        # Create edges based on dependencies or transformations
        for read_access_dict in comp_dict["accesses"]:
            edge = {
                "source": comp_index,
                "target": read_access_dict["buffer_id"],  # Assuming buffer_id indicates a target node
                "type": "read"  # You can differentiate edge types if needed
            }
            edges.append(edge)

    # Create a representation of the loops independently from the computations
    for loop_index, loop_name in enumerate(program_json["iterators"]):
        loop_node = {
            "name": loop_name,
            "index": loop_index,
            "transformations": {
                "parallelized": False,
                "tiled": False,
                "unrolled": False
            }
        }
        nodes.append(loop_node)

    # Create a mapping for node indices
    node_indices_dict = {node["name"]: idx for idx, node in enumerate(nodes)}

    # Build graph structure
    graph = {
        "nodes": nodes,
        "edges": edges,
        "node_indices": node_indices_dict
    }

    # Get the original version of the program 
    no_sched_json = program_dict["schedules_list"][0]
    
    # Make sure no fusion was applied on this version and get the original tree structure 
    assert "fusions" not in no_sched_json or no_sched_json["fusions"] is None
    orig_tree_structure = no_sched_json["tree_structure"]
    tree_annotation = orig_tree_structure.copy()
    
    # Add necessary attributes to the tree_structure
    prog_tree = update_tree_atributes(tree_annotation, node_indices_dict, train_device=train_device)

    return graph, prog_tree
# Change the structure of the tree annotations to contain a uinque index for each loop and a has_comps boolean
# This is used to prepare for the recusive embedding of the program during the training
def update_tree_attributes(node, graph, train_device="cpu"):
    """
    Updates the tree attributes for use in a GNN by adding node indices and establishing 
    hierarchical relationships in the graph structure.
    
    Args:
        node: Current node in the tree
        graph: Graph representation containing nodes, edges, and indices
        train_device: Device to place tensors on
    """
    if "roots" in node:
        # Handle multiple roots if present
        for root in node["roots"]:
            update_tree_attributes(root, graph, train_device=train_device)
        return node

    # Get node index from graph
    if "loop_name" in node:
        # For loop nodes
        node["loop_index"] = torch.tensor(
            graph["node_indices"][node["loop_name"]]
        ).to(train_device)
        
        # Add hierarchical edges to graph
        for child_node in node["child_list"]:
            if "loop_name" in child_node:
                # Add parent-child edge between loops
                graph["edges"].append({
                    "source": graph["node_indices"][node["loop_name"]],
                    "target": graph["node_indices"][child_node["loop_name"]],
                    "type": "parent_child"
                })

    # Handle computations in the node
    if node["computations_list"]:
        # Create computation indices tensor
        node["computations_indices"] = torch.tensor([
            graph["node_indices"][comp_name]
            for comp_name in node["computations_list"]
        ]).to(train_device)
        
        # Add edges between loop and its computations
        for comp_name in node["computations_list"]:
            graph["edges"].append({
                "source": graph["node_indices"][node["loop_name"]],
                "target": graph["node_indices"][comp_name],
                "type": "loop_computation"
            })
        
        node["has_comps"] = True
    else:
        node["has_comps"] = False

    # Recursively update children
    for child_node in node["child_list"]:
        update_tree_attributes(child_node, graph, train_device=train_device)

    return node

    
    
# Fill the representation template with the corresponsing schedule features 
def get_schedule_representation(
    program_json,
    schedule_json,
    graph,
    train_device="cpu"
):
    """
    Creates a graph representation of a schedule including transformations
    """
    # Get computations dictionary and order
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )

    # Create node features and edge features tensors
    node_features = []
    edge_features = []
    
    # For each computation in the program
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        comp_schedule_dict = schedule_json[comp_name]
        
        # Get the computation node index from graph
        comp_node_idx = graph["node_indices"][comp_name]
        
        # Initialize node features for this computation
        comp_features = {
            "is_reduction": comp_dict["comp_is_reduction"],
            "write_buffer_id": comp_dict["write_buffer_id"] + 1,
            "expression": get_tree_expr_repr(comp_dict["expression_representation"], comp_dict["data_type"])
        }

        # Handle transformations
        transformation_features = {
            "parallelized": False,
            "tiled": False,
            "tile_factor": 0,
            "unrolled": False,
            "unroll_factor": 0,
            "fused": False,
            "shifted": False,
            "shift_factor": 0
        }

        # Process iterator transformations
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            # Check parallelization
            if iterator_name == comp_schedule_dict["parallelized_dim"]:
                transformation_features["parallelized"] = True

            # Check tiling
            if comp_schedule_dict["tiling"] and (
                iterator_name in comp_schedule_dict["tiling"]["tiling_dims"]
            ):
                transformation_features["tiled"] = True
                tile_factor_index = comp_schedule_dict["tiling"]["tiling_dims"].index(
                    iterator_name
                )
                transformation_features["tile_factor"] = int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tile_factor_index]
                )

            # Check shifting
            if comp_schedule_dict['shiftings']:
                for shifting in comp_schedule_dict['shiftings']:
                    if iterator_name.startswith(shifting[0]):
                        transformation_features["shifted"] = True
                        transformation_features["shift_factor"] = shifting[1]
                        break

        # Check unrolling
        if comp_schedule_dict["unrolling_factor"]:
            transformation_features["unrolled"] = True
            transformation_features["unroll_factor"] = int(comp_schedule_dict["unrolling_factor"])

        # Combine features
        node_features.append({**comp_features, **transformation_features})

        # Handle fusion by creating edges
        if "fusions" in schedule_json and schedule_json["fusions"]:
            for fusion in schedule_json["fusions"]:
                if comp_name in fusion:
                    # Create fusion edges
                    for other_comp in fusion:
                        if other_comp != comp_name:
                            edge_features.append({
                                "source": comp_node_idx,
                                "target": graph["node_indices"][other_comp],
                                "type": "fusion",
                                "level": fusion[2]  # fusion level
                            })

    # Convert features to tensors
    node_features_tensor = torch.tensor([
        [
            float(feat[key]) for key in sorted(feat.keys())
        ] for feat in node_features
    ]).to(train_device)

    edge_index = torch.tensor([
        [edge["source"], edge["target"]] for edge in edge_features
    ], dtype=torch.long).t().to(train_device)

    edge_attr = torch.tensor([
        [
            float(edge["type"] == "fusion"),
            float(edge["level"]) if "level" in edge else 0.0
        ] for edge in edge_features
    ]).to(train_device)

    return node_features_tensor, edge_index, edge_attr


# Get the representation of a specific set of functions and write it to a pkl file for the parent process to read
def get_func_repr_task(input_q, output_q):
    """
    Process function representations for GNN in parallel
    """
    # Receive functions to work on from parent process
    process_id, programs_dict, repr_pkl_output_folder, train_device = input_q.get()
    function_name_list = list(programs_dict.keys())
    dropped_funcs = []
    local_list = []

    for function_name in tqdm(function_name_list):
        nb_dropped = 0
        
        # Check whether this function should be dropped
        if drop_program(programs_dict[function_name], function_name):
            dropped_funcs.append(function_name)
            continue
            
        # Get the JSON representation of the program features
        program_json = programs_dict[function_name]["program_annotation"]
        
        # Extract the graph representation template for the datapoint
        try:
            (
                graph,
                prog_tree,
            ) = get_graph_representation_template(
                programs_dict[function_name],
                train_device=train_device,
            )
        
        except (LoopsDepthException, NbAccessException):
            # If one of the two exceptions was raised, we drop all the schedules for that program
            nb_dropped += len(programs_dict[function_name]["schedules_list"])
            continue

        # Get the initial execution time for speedup calculation
        program_exec_time = programs_dict[function_name]["initial_execution_time"]
        
        # Get the program tree footprint
        tree_footprint = get_tree_footprint(prog_tree)
        
        # Initialize local storage for this function
        local_function_dict = {
            "graph": graph,
            "tree": prog_tree,
            "node_features_list": [],
            "edge_index_list": [],
            "edge_attr_list": [],
            "datapoint_attributes_list": [],
            "speedups_list": [],
            "exec_time_list": [],
            "func_id": [],
        }

        # Process each schedule for this function
        for schedule_index in range(len(programs_dict[function_name]['schedules_list'])):
            # Get the schedule JSON representation
            schedule_json = programs_dict[function_name]['schedules_list'][schedule_index]
            
            # Get the transformed execution time
            sched_exec_time = np.min(schedule_json['execution_times'])
            
            # Check if this schedule should be dropped
            if drop_schedule(programs_dict[function_name], schedule_index) or (not sched_exec_time):
                nb_dropped += 1
                continue
                
            # Calculate speedup
            sched_speedup = program_exec_time / sched_exec_time
            sched_speedup = speedup_clip(sched_speedup)
            
            try:
                # Get graph representation for this schedule
                node_features, edge_index, edge_attr = get_schedule_representation(
                    program_json,
                    schedule_json,
                    graph,
                    train_device=train_device,
                )

            except NbTranformationException:
                # Skip if number of transformations exceeds maximum
                nb_dropped += 1 
                continue
                
            # Get datapoint attributes
            datapoint_attributes = get_datapoint_attributes(
                function_name, 
                programs_dict[function_name], 
                schedule_index, 
                tree_footprint
            )
            
            # Store features and attributes
            local_function_dict['node_features_list'].append(node_features)
            local_function_dict['edge_index_list'].append(edge_index)
            local_function_dict['edge_attr_list'].append(edge_attr)
            local_function_dict['datapoint_attributes_list'].append(datapoint_attributes)
            local_function_dict['speedups_list'].append(sched_speedup)
            
        # Add the function information to the output list
        local_list.append((
            function_name, 
            nb_dropped, 
            tree_footprint, 
            local_function_dict
        ))
        
    # Write output representation to pkl files
    pkl_part_filename = repr_pkl_output_folder + '/pickled_representation_part_'+str(process_id)+'.pkl'
    with open(pkl_part_filename, 'wb') as f:
        pickle.dump(local_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Send file path to parent process
    output_q.put((process_id, pkl_part_filename))
# A class to contain the training and validation datasets
# Parameters:
#     dataset_filename: path to training/val dataset
#     max_batch_size: maximum batch size
#     drop_sched_func: function specifying which schedules in the dataset we want to be dropped if any
#     drop_prog_func: function specifying which programs in the dataset we want to be dropped if any
#     speedups_clip_func: function spesifying which speedups to clip and to what value
#     store_device: device where to store the dataset
#     train_device: device where to train the model
class Dataset_parallel:
    def __init__(
            self,
            dataset_filename,
            max_batch_size=1024,
            drop_sched_func=None,
            drop_prog_func=None,
            speedups_clip_func=None,
            no_batching=False,
            store_device="cpu",
            train_device="cpu",
            repr_pkl_output_folder="none",
            just_load_pickled_repr=False,
            nb_processes=15,
            min_functions_per_tree_footprint=0
        ):
        # Initialize data structures
        self.batched_graphs = []  # Store batched graph data
        self.batched_labels = []  # Store corresponding labels
        self.nb_dropped = 0       # Number of dropped schedules
        self.nb_pruned = 0        # Number of pruned schedules
        self.dropped_funcs = []   # List of dropped functions
        self.batched_datapoint_attributes = []  # Metadata for analysis
        self.nb_datapoints = 0    # Total number of datapoints
        self.gpu_fitted_batches_index = -1  # Track GPU memory limit
        self.nb_funcs_per_footprint = {}   # Functions per tree structure

        processes_output_list = []
        programs_dict = {}
        batches_dict = dict()

        # Handle loading from pickled files
        if just_load_pickled_repr:
            for pkl_part_filename in tqdm(list(Path(repr_pkl_output_folder).iterdir())):
                with open(str(pkl_part_filename), 'rb') as f:
                    processes_output_list.extend(pickle.load(f))
        else:
            # Set up parallel processing
            manager = multiprocessing.Manager()
            processes = []
            input_queue = manager.Queue()
            output_queue = manager.Queue()

            # Start worker processes
            for i in range(nb_processes):
                processes.append(multiprocessing.Process(
                    target=get_func_repr_task, 
                    args=[input_queue, output_queue]
                ))
                processes[-1].start()

            # Load and distribute dataset
            if dataset_filename.endswith("json"):
                with open(dataset_filename, "r") as f:
                    programs_dict = json.loads(f.read())
            elif dataset_filename.endswith("pkl"):
                with open(dataset_filename, "rb") as f:
                    programs_dict = pickle.load(f)

            # Distribute work among processes
            functions_list = list(programs_dict.keys())
            random.Random(42).shuffle(functions_list)
            nb_funcs_per_process = (len(functions_list)//nb_processes)+1

            for i in range(nb_processes):
                process_programs_dict = dict(
                    list(programs_dict.items())[i*nb_funcs_per_process:(i+1)*nb_funcs_per_process]
                )
                input_queue.put((i, process_programs_dict, repr_pkl_output_folder, store_device))

            # Collect results
            for i in range(nb_processes):
                process_id, pkl_part_filename = output_queue.get()
                if not no_batching:
                    with open(pkl_part_filename, 'rb') as f:
                        processes_output_list.extend(pickle.load(f))

        # Set default functions if not provided
        if drop_sched_func is None:
            drop_sched_func = lambda x, y: False
        if drop_prog_func is None:
            drop_prog_func = lambda x, y: False
        if speedups_clip_func is None:
            speedups_clip_func = lambda x: x

        if no_batching:
            print("Parameter no_batching is True. Stopping after PKL files were saved.")
            return

        print("Assembling schedules from each function")
        # Process each function's data
        for function_name, nb_dropped, tree_footprint, local_function_dict in processes_output_list:
            # Update statistics
            self.nb_dropped += nb_dropped
            self.nb_datapoints += len(local_function_dict['speedups_list'])
            
            # Track functions per footprint
            if tree_footprint not in self.nb_funcs_per_footprint:
                self.nb_funcs_per_footprint[tree_footprint] = {'nb_funcs': 0, 'nb_dps': 0}
            self.nb_funcs_per_footprint[tree_footprint]['nb_funcs'] += 1
            self.nb_funcs_per_footprint[tree_footprint]['nb_dps'] += len(local_function_dict['speedups_list'])

            # Initialize batch dictionary for new footprints
            if tree_footprint not in batches_dict:
                batches_dict[tree_footprint] = {
                    'graph_template': local_function_dict['graph'],
                    'node_features': [],
                    'edge_indices': [],
                    'edge_attrs': [],
                    'datapoint_attributes': [],
                    'speedups': []
                }

            # Add data to batches
            batch = batches_dict[tree_footprint]
            batch['node_features'].extend(local_function_dict['node_features'])
            batch['edge_indices'].extend(local_function_dict['edge_indices'])
            batch['edge_attrs'].extend(local_function_dict['edge_attrs'])
            batch['datapoint_attributes'].extend(local_function_dict['datapoint_attributes'])
            batch['speedups'].extend(local_function_dict['speedups'])

        # Clean up
        del processes_output_list, programs_dict
        gc.collect()

        print("Creating batches")
        storing_device = torch.device(store_device)

        # Process each tree footprint
        for tree_footprint in tqdm(batches_dict):
            # Skip if insufficient data
            if (self.nb_funcs_per_footprint[tree_footprint]['nb_funcs'] < min_functions_per_tree_footprint and 
                self.nb_funcs_per_footprint[tree_footprint]['nb_dps'] < 100 * min_functions_per_tree_footprint):
                self.nb_datapoints -= self.nb_funcs_per_footprint[tree_footprint]['nb_dps']
                continue

            batch_data = batches_dict[tree_footprint]
            
            # Create batches of size max_batch_size
            for i in range(0, len(batch_data['speedups']), max_batch_size):
                # Check GPU memory
                if storing_device.type == "cuda" and (
                    torch.cuda.memory_allocated(storing_device.index) / 
                    torch.cuda.get_device_properties(storing_device.index).total_memory > 0.83
                ):
                    print(f"GPU memory on {storing_device} nearly full, switching to CPU memory")
                    self.gpu_fitted_batches_index = len(self.batched_graphs)
                    storing_device = torch.device("cpu")

                end_idx = min(i + max_batch_size, len(batch_data['speedups']))
                
                # Create graph batch
                graph_batch = {
                    'node_features': torch.stack(batch_data['node_features'][i:end_idx]).to(storing_device),
                    'edge_index': torch.stack(batch_data['edge_indices'][i:end_idx]).to(storing_device),
                    'edge_attr': torch.stack(batch_data['edge_attrs'][i:end_idx]).to(storing_device),
                    'graph_template': batch_data['graph_template']
                }

                self.batched_graphs.append(graph_batch)
                self.batched_labels.append(
                    torch.FloatTensor(batch_data['speedups'][i:end_idx]).to(storing_device)
                )
                self.batched_datapoint_attributes.append(
                    batch_data['datapoint_attributes'][i:end_idx]  )

# Function to read the pkls written by the load_data_into_pkls_parallel function, batch the loaded data and return the batched data to be saved
def load_pickled_repr(
    repr_pkl_output_folder=None,
    max_batch_size=1024, 
    store_device="cpu", 
    train_device="cpu", 
    min_functions_per_tree_footprint=0
):
    """
    Load pickled representations and create GNN-compatible dataset
    
    Args:
        repr_pkl_output_folder: Path to pickled representations
        max_batch_size: Maximum batch size for processing
        store_device: Device to store the data (cpu/cuda)
        train_device: Device for training
        min_functions_per_tree_footprint: Minimum functions required per tree structure
        
    Returns:
        dataset: Dataset_parallel object
        batches_list: List of (graph_batch, labels) tuples
        indices: List of batch indices
        gpu_fitted_batches_index: Index indicating GPU memory limit
    """
    # Initialize dataset with pickled representations
    dataset = Dataset_parallel(
        None, 
        max_batch_size=max_batch_size, 
        None, 
        repr_pkl_output_folder=repr_pkl_output_folder, 
        just_load_pickled_repr=True, 
        store_device=store_device, 
        train_device=train_device,
        min_functions_per_tree_footprint=min_functions_per_tree_footprint
    )
    
    # Create list of indices
    indices = list(range(len(dataset)))
    
    # Create batches list containing (graph_batch, labels) tuples
    batches_list = []
    for i in indices:
        # Get graph batch and labels from dataset
        graph_batch = dataset.batched_graphs[i]
        labels = dataset.batched_labels[i]
        
        # Create tuple of (graph_batch, labels)
        batch_tuple = (graph_batch, labels)
        batches_list.append(batch_tuple)

    return dataset, batches_list, indices, dataset.gpu_fitted_batches_index  

def load_data_into_pkls_parallel(
    train_val_dataset_file, 
    nb_processes=15, 
    repr_pkl_output_folder=None, 
    overwrite_existing_pkl=False
):
    """
    Load data from a dataset file, extract GNN representations, and save them into PKL files.

    Args:
        train_val_dataset_file: Path to the input dataset file (JSON or PKL)
        nb_processes: Number of parallel processes to use for data extraction
        repr_pkl_output_folder: Directory to save the pickled representations
        overwrite_existing_pkl: Flag to overwrite existing PKL files

    Returns:
        None
    """
    # Check if the output folder exists and handle overwriting if necessary
    if Path(repr_pkl_output_folder).is_dir() and overwrite_existing_pkl:
        shutil.rmtree(repr_pkl_output_folder)
        print('Deleted existing folder ', repr_pkl_output_folder)
        
    # Create the output folder for pickled representations
    Path(repr_pkl_output_folder).mkdir(parents=True, exist_ok=False)
    print('Created folder ', repr_pkl_output_folder)
    
    # Read the dataset and write the GNN representation into PKL files
    print("Loading data from: " + train_val_dataset_file)
    dataset = Dataset_parallel(
        train_val_dataset_file,
        no_batching=True,
        just_load_pickled_repr=False,
        nb_processes=nb_processes, 
        repr_pkl_output_folder=repr_pkl_output_folder, 
        store_device="cpu", 
        train_device="cpu"
    )
    return  

# Returns a representation of the tree structure of the program
def get_tree_footprint(tree):
    """
    Generate a unique string representation (footprint) of the tree structure for GNN processing.
    This footprint helps in batching similar graph structures together.
    
    Args:
        tree: Dictionary containing the tree/graph structure with nodes, edges, and computation info
        
    Returns:
        str: A string representation of the tree structure
    """
    footprint = ""
    
    # Handle root nodes if present
    if "roots" in tree:
        for root in tree["roots"]:
            footprint += "<R>"  # Root node marker
            footprint += get_tree_footprint(root)
        footprint += "</R>"
        return footprint
    
    # Add loop index to footprint
    footprint = f"<L{int(tree['loop_index'])}>"
    
    # Add computation information if present
    if tree["has_comps"]:
        footprint += "["
        # Add computation indices in sorted order for consistency
        comp_indices = sorted([int(idx) for idx in tree["computations_indices"]])
        for idx in comp_indices:
            footprint += f"C{idx}"
        footprint += "]"
    
    # Add node connectivity information
    if "edges" in tree:
        footprint += "<E"
        # Sort edge information for consistent footprints
        sorted_edges = sorted(tree["edges"], key=lambda x: (x[0], x[1]))
        for edge in sorted_edges:
            footprint += f"{edge[0]}-{edge[1]}"
        footprint += ">"
    
    # Add node feature information if present
    if "node_features" in tree:
        footprint += "<F" + str(tree["node_features"].shape[-1]) + ">"
    
    # Recursively process child nodes
    for child in tree["child_list"]:
        footprint += get_tree_footprint(child)
    
    # Close the loop tag
    footprint += f"</L{int(tree['loop_index'])}>"
    
    return footprint

# A function to extract the transformations applied on a spesific computation in the form of a vector of tags
# Padding is added if the number of transformations is less than the maximum value of MAX_NUM_TRANSFORMATIONS
# The tag representation is as follows:
#         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop', 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4', 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
#     Where the type_of_transformation tag is:
#         - 0 for no transformation being applied
#         - 1 for loop interchange
#         - 2 for loop reversal
#         - 3 for loop skewing
# In the case for skewing we are specifying the new values for the transformed submatrix
def get_padded_transformation_tags(program_json, schedule_json, comp_name):
    """
    Extract and format transformation information as graph features/attributes.
    
    Args:
        program_json: JSON containing program information
        schedule_json: JSON containing schedule information
        comp_name: Name of the computation
        
    Returns:
        dict: Dictionary containing transformation features for GNN processing
    """
    # Extract information about the computation and transformations
    comp_dict = program_json["computations"][comp_name]
    comp_schedule_dict = schedule_json[comp_name]
    nb_iterators = len(comp_dict["iterators"])
    loop_nest = comp_dict["iterators"][:]
    
    # Initialize transformation features
    transformation_features = {
        'node_features': [],      # Features for each node
        'edge_features': [],      # Features for edges
        'edge_indices': [],       # Connectivity information
        'global_features': []     # Global transformation features
    }
    
    # Create base identity matrix for no transformations
    identity = np.zeros((nb_iterators, nb_iterators), int)
    np.fill_diagonal(identity, 1)
    
    # Process each transformation
    for transformation in comp_schedule_dict['transformations_list']:
        trans_type = transformation[0]
        
        # Node features based on transformation type
        node_feat = np.zeros(MAX_TAGS, dtype=np.float32)
        node_feat[0] = trans_type  # Transformation type
        
        if trans_type == 1:  # Interchange
            # Create edge between interchanged loops
            edge_index = [transformation[1], transformation[2]]
            edge_feat = np.ones(MAX_TAGS // 2, dtype=np.float32)  # Interchange-specific features
            
            transformation_features['edge_indices'].append(edge_index)
            transformation_features['edge_features'].append(edge_feat)
            
        elif trans_type == 2:  # Reversal
            # Add reversal-specific node features
            node_feat[1:4] = [1, transformation[3], 0]  # Reversal flags and index
            
        elif trans_type == 3:  # Skewing
            # Add skewing-specific features and edges
            if transformation[6] == 0:  # 2D skewing
                edge_index = [transformation[4], transformation[5]]
                edge_feat = np.array([transformation[7:11]], dtype=np.float32)  # Skewing factors
                
                transformation_features['edge_indices'].append(edge_index)
                transformation_features['edge_features'].append(edge_feat)
                
            else:  # 3D skewing
                # Create edges for 3D skewing
                edges = [
                    [transformation[4], transformation[5]],
                    [transformation[5], transformation[6]],
                    [transformation[4], transformation[6]]
                ]
                # Skewing factors as edge features
                edge_feats = [
                    np.array([transformation[7:10]], dtype=np.float32),
                    np.array([transformation[10:13]], dtype=np.float32),
                    np.array([transformation[13:16]], dtype=np.float32)
                ]
                
                transformation_features['edge_indices'].extend(edges)
                transformation_features['edge_features'].extend(edge_feats)
        
        transformation_features['node_features'].append(node_feat)
    
    # Add padding for consistent sizes
    num_transforms = len(transformation_features['node_features'])
    if num_transforms < MAX_NUM_TRANSFORMATIONS:
        # Pad node features
        pad_node_feat = np.zeros(MAX_TAGS, dtype=np.float32)
        transformation_features['node_features'].extend(
            [pad_node_feat] * (MAX_NUM_TRANSFORMATIONS - num_transforms)
        )
        
        # Pad edge features if needed
        if transformation_features['edge_features']:
            pad_edge_feat = np.zeros_like(transformation_features['edge_features'][0])
            pad_edge_index = [0, 0]  # Dummy edge
            
            num_edges = len(transformation_features['edge_features'])
            transformation_features['edge_features'].extend(
                [pad_edge_feat] * (MAX_NUM_TRANSFORMATIONS - num_edges)
            )
            transformation_features['edge_indices'].extend(
                [pad_edge_index] * (MAX_NUM_TRANSFORMATIONS - num_edges)
            )
    
    # Convert lists to numpy arrays
    transformation_features['node_features'] = np.array(transformation_features['node_features'])
    transformation_features['edge_features'] = np.array(transformation_features['edge_features']) if transformation_features['edge_features'] else np.array([])
    transformation_features['edge_indices'] = np.array(transformation_features['edge_indices']) if transformation_features['edge_indices'] else np.array([])
    
    # Add global features (e.g., number of transformations, iterator information)
    transformation_features['global_features'] = np.array([
        num_transforms,
        nb_iterators,
        len(transformation_features['edge_indices'])
    ])
    
    return transformation_features

# A function to retrieve information about each datapoint
def get_datapoint_attributes(func_name, program_dict, schedule_index, tree_footprint):
    """
    Extract and format attributes/features of a datapoint for GNN processing.
    
    Args:
        func_name: Name of the function
        program_dict: Dictionary containing program information
        schedule_index: Index of the schedule
        tree_footprint: String representation of the tree structure
        
    Returns:
        tuple: Contains various attributes of the datapoint including graph-specific features
    """
    # Get basic schedule information
    schedule_json = program_dict["schedules_list"][schedule_index]
    sched_id = str(schedule_index).zfill(4)
    sched_str = get_schedule_str(program_dict["program_annotation"], schedule_json)
    
    # Performance metrics
    exec_time = np.min(schedule_json["execution_times"])
    memory_use = program_dict["program_annotation"]["memory_size"]
    speedup = program_dict["initial_execution_time"] / exec_time
    
    # Node/hardware information
    node_name = program_dict["node_name"] if "node_name" in program_dict else "unknown"
    
    # Extract graph-specific attributes
    graph_attributes = {
        # Program structure attributes
        "num_nodes": len(program_dict["program_annotation"]["iterators"]),
        "num_computations": len(program_dict["program_annotation"]["computations"]),
        
        # Memory access patterns
        "memory_access_pattern": extract_memory_pattern(program_dict["program_annotation"]),
        
        # Dependency information
        "dependencies": extract_dependencies(program_dict["program_annotation"]),
        
        # Hardware-specific features
        "hardware_features": extract_hardware_features(program_dict),
        
        # Schedule-specific attributes
        "schedule_features": extract_schedule_features(schedule_json)
    }

    return (
        func_name,           # Function name
        sched_id,           # Schedule ID
        sched_str,          # Schedule string representation
        exec_time,          # Execution time
        memory_use,         # Memory usage
        node_name,          # Node/hardware name
        tree_footprint,     # Tree structure footprint
        speedup,            # Performance speedup
        graph_attributes    # Additional graph-specific attributes
    )

def extract_memory_pattern(program_annotation):
    """
    Extract memory access pattern features for GNN processing.
    
    Args:
        program_annotation: Program annotation dictionary
        
    Returns:
        dict: Memory access pattern features
    """
    memory_pattern = {
        "access_matrices": [],
        "access_types": [],
        "buffer_sizes": []
    }
    
    for comp_name, comp_info in program_annotation["computations"].items():
        # Add write access pattern
        if "write_access_relation" in comp_info:
            memory_pattern["access_matrices"].append(
                isl_to_write_matrix(comp_info["write_access_relation"])
            )
            memory_pattern["access_types"].append("write")
            
        # Add read access patterns
        for access in comp_info.get("accesses", []):
            memory_pattern["access_matrices"].append(access["access_matrix"])
            memory_pattern["access_types"].append("read")
            
        # Add buffer sizes if available
        if "buffer_size" in comp_info:
            memory_pattern["buffer_sizes"].append(comp_info["buffer_size"])
            
    return memory_pattern

def extract_dependencies(program_annotation):
    """
    Extract dependency information for GNN processing.
    
    Args:
        program_annotation: Program annotation dictionary
        
    Returns:
        dict: Dependency features
    """
    dependencies = {
        "data_dependencies": [],
        "control_dependencies": [],
        "dependency_distances": []
    }
    
    # Extract dependencies between computations
    computations = program_annotation["computations"]
    for comp_name, comp_info in computations.items():
        for access in comp_info.get("accesses", []):
            if access.get("access_is_dependency", False):
                dependencies["data_dependencies"].append({
                    "source": access["buffer_id"],
                    "target": comp_info["absolute_order"],
                    "type": "read-after-write"
                })
    
    return dependencies

def extract_hardware_features(program_dict):
    """
    Extract hardware-specific features for GNN processing.
    
    Args:
        program_dict: Program dictionary containing hardware information
        
    Returns:
        dict: Hardware-specific features
    """
    hardware_features = {
        "cache_size": program_dict.get("cache_size", 0),
        "memory_bandwidth": program_dict.get("memory_bandwidth", 0),
        "num_cores": program_dict.get("num_cores", 1),
        "vector_units": program_dict.get("vector_units", 1)
    }
    
    return hardware_features

def extract_schedule_features(schedule_json):
    """
    Extract schedule-specific features for GNN processing.
    
    Args:
        schedule_json: Schedule information dictionary
        
    Returns:
        dict: Schedule-specific features
    """
    schedule_features = {
        "transformation_sequence": [],
        "parallel_loops": [],
        "tiling_info": [],
        "unroll_info": []
    }
    
    # Extract transformation sequence
    for comp_name, comp_schedule in schedule_json.items():
        if isinstance(comp_schedule, dict) and "transformations_list" in comp_schedule:
            schedule_features["transformation_sequence"].extend(
                comp_schedule["transformations_list"]
            )
            
            # Extract parallelization information
            if comp_schedule.get("parallelized_dim"):
                schedule_features["parallel_loops"].append(
                    comp_schedule["parallelized_dim"]
                )
                
            # Extract tiling information
            if comp_schedule.get("tiling"):
                schedule_features["tiling_info"].append(comp_schedule["tiling"])
                
            # Extract unrolling information
            if comp_schedule.get("unrolling_factor"):
                schedule_features["unroll_info"].append({
                    "comp": comp_name,
                    "factor": comp_schedule["unrolling_factor"]
                })
    
    return schedule_features
# Add padding to the read/write access matrices
def pad_access_matrix(access_matrix, max_depth=MAX_DEPTH):
    """
    Pad access matrix for GNN processing with additional structural information.
    
    Args:
        access_matrix: Original access matrix
        max_depth: Maximum depth for padding
    
    Returns:
        dict: Dictionary containing padded matrix and additional features
    """
    access_matrix = np.array(access_matrix)
    
    # Add bias column and row
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    
    # Create padded matrix
    padded_matrix = np.zeros((max_depth + 1, max_depth + 2))
    padded_matrix[: access_matrix.shape[0], : access_matrix.shape[1] - 1] = access_matrix[:, :-1]
    padded_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]
    
    # Create additional features for GNN
    access_features = {
        'padded_matrix': padded_matrix,
        'original_shape': access_matrix.shape,
        'sparsity': np.count_nonzero(access_matrix) / access_matrix.size,
        'access_pattern': {
            'sequential': np.all(np.diff(access_matrix, axis=1) == 1),
            'strided': np.any(np.diff(access_matrix, axis=1) > 1),
            'irregular': not np.all(np.diff(np.sort(access_matrix.flatten())) == 1)
        }
    }
    
    return access_features

# Tranfrom the access relations to matrices
def isl_to_write_matrix(isl_map):
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                break
    return matrix

def isl_to_write_matrix(isl_map):
    """
    Convert ISL map to write matrix with additional GNN-friendly features.
    
    Args:
        isl_map: ISL map string
    
    Returns:
        dict: Dictionary containing matrix and additional features
    """
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    
    # Build connectivity matrix
    connectivity = {}
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                connectivity[buf_iter] = comp_iter
    
    return {
        'matrix': matrix,
        'comp_iterators': comp_iter_names,
        'buffer_iterators': buf_iter_names,
        'connectivity': connectivity,
        'dimension_mapping': {
            'input_dims': len(comp_iter_names),
            'output_dims': len(buf_iter_names)
        }
    }


# returns a pandas dataframe representing the dataset
def get_results_df(dataset, batches_list, indices, model, log=False, train_device="cpu"):
    """
    Get results dataframe with additional GNN-specific metrics.
    """
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    
    # Initialize collectors
    results_collector = {
        'outputs': [],
        'labels': [],
        'graph_metrics': [],
        'attention_weights': [],
        'node_embeddings': [],
        'program_attributes': []
    }
    
    for k, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        
        # Move inputs to training device
        inputs = move_inputs_to_device(inputs, train_device)
        labels = labels.to(train_device)
        
        # Get model predictions and intermediate representations
        with torch.no_grad():
            outputs, attention_weights, node_embeds = model(inputs, return_intermediates=True)
            
        # Collect results
        results_collector['outputs'].append(outputs)
        results_collector['labels'].append(labels)
        results_collector['attention_weights'].append(attention_weights)
        results_collector['node_embeddings'].append(node_embeds)
        
        # Collect program attributes
        zipped_attributes = list(zip(*dataset.batched_datapoint_attributes[indices[k]]))
        results_collector['program_attributes'].append({
            'names': zipped_attributes[0],
            'schedules': zipped_attributes[1],
            'exec_times': zipped_attributes[3],
            'memory_uses': zipped_attributes[4],
            'node_names': zipped_attributes[5],
            'tree_footprints': zipped_attributes[6]
        })
        
        # Move data back to original device
        inputs = move_inputs_to_device(inputs, original_device)
        labels = labels.to(original_device)
    
    # Process collected results
    df = process_results_to_df(results_collector)
    
    # Add GNN-specific metrics
    df = add_gnn_metrics(df)
    
    return df

def move_inputs_to_device(inputs, device):
    """Helper function to move inputs to specified device."""
    return (
        inputs[0],
        inputs[1].to(device),
        inputs[2].to(device),
        inputs[3].to(device),
        inputs[4].to(device),
        inputs[5].to(device),
    )

def process_results_to_df(results_collector):
    """Process collected results into DataFrame."""
    df = pd.DataFrame()
    
    # Process predictions and targets
    preds = torch.cat(results_collector['outputs']).cpu().detach().numpy().reshape((-1,))
    targets = torch.cat(results_collector['labels']).cpu().detach().numpy().reshape((-1,))
    
    # Create base DataFrame
    df["name"] = [item for batch in results_collector['program_attributes'] for item in batch['names']]
    df["tree_struct"] = [item for batch in results_collector['program_attributes'] for item in batch['tree_footprints']]
    df["sched_name"] = [item for batch in results_collector['program_attributes'] for item in batch['schedules']]
    df["exec_time"] = [item for batch in results_collector['program_attributes'] for item in batch['exec_times']]
    df["memory_use"] = [float(item) for batch in results_collector['program_attributes'] for item in batch['memory_uses']]
    df["node_name"] = [item for batch in results_collector['program_attributes'] for item in batch['node_names']]
    df["prediction"] = np.around(preds, decimals=6)
    df["target"] = np.around(targets, decimals=6)
    df["APE"] = np.abs(df.target - df.prediction) / df.target * 100
    
    return df

def add_gnn_metrics(df):
    """Add GNN-specific metrics to DataFrame."""
    # Add attention analysis
    df["attention_score"] = compute_attention_scores(df)
    
    # Add node embedding analysis
    df["node_embedding_similarity"] = compute_embedding_similarity(df)
    
    # Add graph structure metrics
    df["graph_density"] = compute_graph

# Calculate the Normalized Discounted Cumulative Gain while only considiring the top rated schedule (k=1)
def function_wise_ndcg_1(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_1=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=1)
    return pd.Series(dict(nDCG_1=score))

# Calculate the Normalized Discounted Cumulative Gain while only considiring the top 5 rated schedules (k=5)
def function_wise_ndcg_5(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_5=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=5)
    return pd.Series(dict(nDCG_5=score))

# Calculate the Normalized Discounted Cumulative Gain while considiring all the schedules
def function_wise_ndcg_full(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=None)
    return pd.Series(dict(nDCG=score))

# Calculate the Spearman correlation coefficient
def function_wise_spearman(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(Spearman_r=np.nan))
    score = spearmanr(g["target"], g["prediction"])[0]
    return pd.Series(dict(Spearman_r=score))

# Calculate the absolute percentage error
def function_wise_ape(g):
    score = np.mean(g["APE"])
    return pd.Series(dict(MAPE=score))

# calculates the model scores from the dataframe
def get_scores(df):
    with tqdm(total=6) as pbar:
        df_spearman = df.groupby("name").apply(function_wise_spearman).reset_index()
        pbar.update(1)
        df_mape = df.groupby("name").apply(function_wise_ape).reset_index()
        pbar.update(1)
        df_ndcg = df.groupby("name").apply(function_wise_ndcg_full).reset_index()
        pbar.update(1)
        df_ndcg1 = df.groupby("name").apply(function_wise_ndcg_1).reset_index()
        pbar.update(1)
        df_ndcg5 = df.groupby("name").apply(function_wise_ndcg_5).reset_index()
        pbar.update(1)
        df_count = df.groupby("name").agg("count").reset_index()[["name", "sched_name"]]
        df_count.columns = ["name", "count"]
        pbar.update(1)

    scores_df = (
        df_count.merge(df_ndcg, on="name")
        .merge(df_ndcg5, on="name")
        .merge(df_ndcg1, on="name")
        .merge(df_spearman, on="name")
        .merge(df_mape, on="name")
    )
    return scores_df

# Solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
# Used to get skewing parameters
def linear_diophantine_default(f_i, f_j):
    n1 = abs(f_i)
    n2 = abs(f_j)
    
    while(n1 != n2):
        if(n1 > n2):
            n1 -=  n2
        else:
            n2 -=  n1
            
    # Update f_i and f_j to equivalent but prime between themselfs value
    f_i = f_i / n1
    f_j = f_j / n1
    
    found = False
    gamma = 0
    sigma = 1
    
    if (f_j == 1) or (f_i == 1):
        gamma = f_i - 1
        sigma = 1
        # Since sigma = 1  then
        # f_i - gamma * f_j = 1 & using the previous condition :
        #  - f_i = 1 : then gamma = 0 (f_i-1) is enough
        #  - f_j = 1 : then gamma = f_i -1  
    else:
        if (f_j == -1) and (f_i > 1):
            gamma = 1
            sigma = 0
        else:
            # General case : solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
            i = 0
            while (i < 100) and (not found):
                if ((sigma * f_i) % abs(f_j)) == 1:
                    found = True
                else:
                    sigma += 1
                    i += 1
            if not found:
                # Detect infinite loop and prevent it in case where f_i and f_j are not prime between themselfs
                print("Error cannof find solution to diophantine equation")
                return
            gamma = ((sigma * f_i) - 1) / f_j
    return gamma, sigma

# Set a lower bound for speedups to avoid fluctuations and make the training easier
def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup

# Check if this program should be dropped
def drop_program(prog_dict, prog_name):
    # If there are no schedules explored for this function
    if len(prog_dict["schedules_list"]) < 2:
        return True
    
    return False

# Check if this schedule should be dropped
def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    # If the execution list is empty or it contains incoherent executions 
    if (not schedule_json["execution_times"]) or min(schedule_json["execution_times"]) < 0: 
        return True
    
    return False

# Get the involved computations from a specific node 
def get_involved_comps(node):
        result = []
        if(len(node)==0): 
            return result
        for comp in node["computations_list"]:
            result.append(comp)
        for child in node["child_list"]:
            for comp in get_involved_comps(child):
                result.append(comp)
        return result

# Retrieve the iterators that involve this computation from the schedule tree_structure
def get_comp_iterators_from_tree_struct(schedule_json, comp_name):
    tree = schedule_json["tree_structure"]
    level = tree
    iterators = []
    to_explore = []
    # only add the root that contains the computation we are looking for
    for root in tree["roots"]:
        if (comp_name in get_involved_comps(root)):
            to_explore.append(root)
    
    while(to_explore):
        level = to_explore.pop(0)
        if(comp_name in get_involved_comps(level)):
            iterators.append(level['loop_name'])
            
        for element in level["child_list"]:
            to_explore.append(element)
    
    return iterators

# One-hot encoding for expressions and their datatypes
def get_expr_repr(expr, comp_type):
        expr_vector = []
        if(expr == "add"):
            expr_vector = [1, 0, 0, 0, 0, 0, 0, 0]
        elif(expr == "sub"):
            expr_vector = [0, 1, 0, 0, 0, 0, 0, 0]
        elif(expr == "mul"):
            expr_vector = [0, 0, 1, 0, 0, 0, 0, 0]
        elif(expr == "div"):
            expr_vector = [0, 0, 0, 1, 0, 0, 0, 0]
        elif(expr == "sqrt"):
            expr_vector = [0, 0, 0, 0, 1, 0, 0, 0]
        elif(expr == "min"):
            expr_vector = [0, 0, 0, 0, 0, 1, 0, 0]
        elif(expr == "max"):
            expr_vector = [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            expr_vector = [0, 0, 0, 0, 0, 0, 0, 1]
        
        comp_type_vector = []
        if(comp_type == "int32"):
            comp_type_vector = [1, 0, 0]
        elif(comp_type == "float32"):
            comp_type_vector = [0, 1, 0]
        elif(comp_type == "float64"):
            comp_type_vector = [0, 0, 1]
            
        return expr_vector + comp_type_vector
    
# Get the representation of the whole expression recursively
def get_tree_expr_repr(node, comp_type):
        expr_tensor = []
        if node["children"] != []:
            for child_node in node["children"]:
                expr_tensor.extend(get_tree_expr_repr(child_node, comp_type))
        expr_tensor.append(get_expr_repr(node["expr_type"], comp_type))

        return expr_tensor


# A constraint matrix is the set of linear inequalities that describes the iteration domain.
# Example:
# if the iteration domain D is the follwoing
#     {i >= 0
#      i < 128
# D =  j >= 0
#      j < 32
#      k >= 0
#      k < 64}
# The iterator vector is 
# x = [i, 
#      j, 
#      k]
# The coeffcients matrix A would be:
#     [-1,   0,   0,
#       1,   0,   0,
# A=    0,  -1,   0,
#       0,   1,   0,
#       0,   0,  -1,
#       0,   0,   1]
# The second hand side of the equation b (constants vector) is the vector:
#     b= [0,
#         127,
#         0,
#         31,
#         0,
#         63]
# Since:
#    D = Ax<=b
# Get the matrix describing the initial constraints for this program
# EDIT: Padding is done with zeros, this idea is no longer in use. ignore the following comment. --- For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner. The constants vector is padded with zeros
def get_padded_initial_iteration_domain(program_json,comp_name, pad=True):
    #supports bounds of type ± i ± j ±...± cst and max(cst, iter)
    comp_dict = program_json['computations'][comp_name] 
    nb_dims = len(comp_dict['iterators'])
    
    coeff_mat = np.zeros((nb_dims*2,nb_dims),int)
    constants_col = np.zeros((nb_dims*2),int)
    
    for i,iterator_name_row in enumerate(comp_dict['iterators']):# rows loop
        
        # set the diagonal-ish values
        coeff_mat[i*2,i]=-1
        coeff_mat[i*2+1,i]=1
        
        iterator_row_dict = program_json['iterators'][iterator_name_row]
        
        #get the simplified expression of the bounds
        upper_bound = str(sympy.simplify(iterator_row_dict['upper_bound'])).replace(' ','')
        lower_bound = str(sympy.simplify(iterator_row_dict['lower_bound'])).replace(' ','')
        
        # in some special cases, the lower bound of i appears as max(cst, iter) when iter<=i. The following is a non-general solution
        if 'max' in lower_bound or 'Max' in lower_bound:
            lower_bound = re.findall('[mM]ax\(.+,(.+)\)',lower_bound)[0]
        
        # find the iterator names and constants used in the bounds
        iterators_in_upper = re.findall('[a-zA-Z]\w*', upper_bound)
        constants_in_upper = re.findall('(?:^|\+|-)\d+', upper_bound)
        iterators_in_lower = re.findall('[a-zA-Z]\w*', lower_bound)
        constants_in_lower = re.findall('(?:^|\+|-)\d+', lower_bound)
        
        #if no constants used, set to 0
        if not constants_in_upper:
            constants_in_upper = [0]
        else:
            assert len(constants_in_upper)==1
       
        if not constants_in_lower:
            constants_in_lower = [0]
        else:
            assert len(constants_in_lower)==1
        
        #for each iterator in the bounds expression, set the corresponding values in the matrix
        for iter_name in iterators_in_upper:
            col_idx = comp_dict['iterators'].index(iter_name)
            if '-'+iter_name in upper_bound:
                coeff_mat[i*2+1,col_idx]=1
            else:
                coeff_mat[i*2+1,col_idx]=-1
        constants_col[i*2+1]= int(constants_in_upper[0])-1 #adding a -1 because we are representing a non-strict inequality
        
        for iter_name in iterators_in_lower:
            col_idx = comp_dict['iterators'].index(iter_name)
            if '-'+iter_name in lower_bound:
                coeff_mat[i*2,col_idx]=-1
            else:
                coeff_mat[i*2,col_idx]=+1
        constants_col[i*2]= -int(constants_in_lower[0])
                
#     constants_col = constants_col.reshape(-1,1)
    
    
    # Add padding if requested
    if pad:
        padded_coeff_mat = np.pad(coeff_mat, [(0,MAX_DEPTH*2-nb_dims*2),(0,MAX_DEPTH-nb_dims)], mode='constant', constant_values=0)
        
#         #Edit: this idea has been dropped.!! For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner
#         for i in range(nb_dims,MAX_DEPTH):
#             padded_coeff_mat[i*2,i]=-1
#             padded_coeff_mat[i*2+1,i]=1
            
        padded_constants_col = np.pad(constants_col, [(0,MAX_DEPTH*2-nb_dims*2)], mode='constant', constant_values=0)
        return padded_coeff_mat, padded_constants_col
    else:
        return coeff_mat, constants_col
    
# Get the matrix describing the iteration domain after applying a sequence of affine transformations
# The transformed constraint matrix is: the original constraint matrix multiplied by the inverse of the transformation matrix
def get_padded_transformed_iteration_domain(program_json, schedule_json, comp_name, pad=True):
    # Extract the transformations matrix for this schedule
    transformation_matrix = get_transformation_matrix(program_json, schedule_json, comp_name)
    
    # Create the initial constraint matrix without any padding
    A,b = get_padded_initial_iteration_domain(program_json,comp_name, pad=False)
    nb_dims = A.shape[1]
        
    # Get the inverse of the transformation matrix
    inverse = np.linalg.inv(transformation_matrix)
    
    # Multiply thw two to gte the transformed constraint matrix
    result = np.matmul(A, inverse)
    result = np.array(result)
    
    if pad:
        result = np.pad(result, [(0, (MAX_DEPTH)*2 - result.shape[0]), (0, MAX_DEPTH - result.shape[1])],
        mode="constant",
        constant_values=0)
         #EDIT: Padding is done with zeros, this idea is no longer in use. ignore the following comment. ---   For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner
#         for i in range(nb_dims,MAX_DEPTH):
#             result[i*2,i]=-1
#             result[i*2+1,i]=1
        
    return result


# Convert a tags vector describing an affine transfromation (Reversal, Skewing, Interchange) into a matrix that represents the same transformation
def get_trasnformation_matrix_from_vector(transformation, matrix_size):
    matrix = np.identity(matrix_size)
    assert(len(transformation) == MAX_TAGS)
    if (transformation[0] == 1):
        # Interchange
        assert(transformation[1] < matrix_size and transformation[2] < matrix_size)
        matrix[transformation[1], transformation[2]] = 1
        matrix[transformation[1], transformation[1]] = 0
        matrix[transformation[2], transformation[1]] = 1
        matrix[transformation[2], transformation[2]] = 0

    elif (transformation[0] == 2):
        # Reversal
        assert(transformation[3] < matrix_size)
        matrix[transformation[3], transformation[3]] = -1

    elif transformation[0] == 3:
        # 2D Skewing
        if transformation[6] == 0:
            
            assert(transformation[4] < matrix_size and transformation[5] < matrix_size)
            matrix[transformation[4], transformation[4]] = transformation[7]
            matrix[transformation[4], transformation[5]] = transformation[8]
            matrix[transformation[5], transformation[4]] = transformation[9]
            matrix[transformation[5], transformation[5]] = transformation[10]
        if transformation[6] > 0:
            # 3D skeweing
            assert(transformation[4] < matrix_size and transformation[5] < matrix_size and transformation[6] < matrix_size)
            matrix[transformation[4], transformation[4]] = transformation[7]
            matrix[transformation[4], transformation[5]] = transformation[8]
            matrix[transformation[4], transformation[6]] = transformation[9]
            matrix[transformation[5], transformation[4]] = transformation[10]
            matrix[transformation[5], transformation[5]] = transformation[11]
            matrix[transformation[5], transformation[6]] = transformation[12]
            matrix[transformation[6], transformation[4]] = transformation[13]
            matrix[transformation[6], transformation[5]] = transformation[14]
            matrix[transformation[6], transformation[6]] = transformation[15]
        
    return matrix

# Transform a sequence of transformation vectors into a single transfromation matrix that represents the whole sequence
def get_transformation_matrix(
    program_json, schedule_json, comp_name
):
    nb_iterators = len(program_json["computations"][comp_name]["iterators"])
    final_transformation = np.identity(nb_iterators)
    for transformation in schedule_json[comp_name]["transformations_list"]:
        matrix = get_trasnformation_matrix_from_vector(transformation, nb_iterators)
        final_transformation = np.matmul(matrix, final_transformation)
    return final_transformation

# Returns a string representation of a schedule and the transformations applied in it
def get_schedule_str(program_json, sched_json):
    comp_name = [
        n
        for n in sched_json.keys()
        if not n in ["unfuse_iterators", "tree_structure", "execution_times", "fusions", "sched_str", "legality_check", "exploration_method"]
    ]
    sched_str = ""
    
    if ("fusions" in sched_json and sched_json["fusions"]):
        for fusion in sched_json["fusions"]:
            sched_str += "F("
            for name in comp_name:
                if name in fusion:
                    sched_str += name + ","
            
            sched_str = sched_str[:-1]
            sched_str += ")"
            
    for name in comp_name:
        transf_loop_nest = program_json["computations"][name]["iterators"].copy()
        schedule = sched_json[name]
        if ("fusions" in sched_json and sched_json["fusions"]):
            for fusion in sched_json["fusions"]:
                # if this computation was involved in a fusion, we know it uses the same iterators as the computation it was fused with
                if name in fusion:
                    iterator_comp_name = fusion[0]
                    transf_loop_nest = program_json["computations"][iterator_comp_name]["iterators"].copy()
                    schedule = sched_json[iterator_comp_name]
        # Change fusion to include loops
        sched_str += '{' + name + '}:'

        for transformation in schedule["transformations_list"]:

            if (transformation[0] == 1):
                sched_str += "I(L" + str(transformation[1]) + ",L" + str(transformation[2]) + ")"
                # Change loop nest to reflect interchange
                assert(transformation[1] < len(transf_loop_nest) and transformation[2] < len(transf_loop_nest))
                tmp_it = transf_loop_nest[transformation[1]]
                transf_loop_nest[transformation[1]] = transf_loop_nest[transformation[2]]
                transf_loop_nest[transformation[2]] = tmp_it
                
            elif (transformation[0] == 2):
                sched_str += "R(L" + str(transformation[3])+ ")"
            elif (transformation[0] == 3):
                sched_str += "S(L" + str(transformation[4]) + ",L" + str(transformation[5]) + "," + str(transformation[6]) + "," + str(transformation[7]) + ")"
                
        if schedule["parallelized_dim"]:
            dim_index = transf_loop_nest.index(schedule["parallelized_dim"])
            sched_str += "P(L" + str(dim_index) + ")"
            
        if schedule["shiftings"]:    
            for shifting in schedule['shiftings']: 
                dim_index = transf_loop_nest.index(shifting[0])
                sched_str += "Sh(L" + str(dim_index) + "," + str(shifting[1])+")"
                
        if schedule["tiling"]:
            if schedule["tiling"]["tiling_depth"] == 1:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                
                first_dim_index = transf_loop_nest.index(first_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                sched_str += (
                    "T1(L"
                    + str(first_dim_index)
                    + ","
                    + str(first_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_outer", first_dim + "_inner"
            elif schedule["tiling"]["tiling_depth"] == 2:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                sched_str += (
                    "T2(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_outer", second_dim + "_outer"
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_inner", second_dim + "_inner"
            elif schedule["tiling"]["tiling_depth"] == 3:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                third_dim = schedule["tiling"]["tiling_dims"][2]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                third_dim_index = transf_loop_nest.index(third_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                third_factor = schedule["tiling"]["tiling_factors"][2]
                sched_str += (
                    "T3(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ",L"
                    + str(third_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ","
                    + str(third_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_outer",
                    second_dim + "_outer",
                    third_dim + "_outer",
                )
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_inner",
                    second_dim + "_inner",
                    third_dim + "_inner",
                )
                transf_loop_nest.remove(third_dim)

        if schedule["unrolling_factor"]:
            dim_index = len(transf_loop_nest) - 1
            dim_name = transf_loop_nest[-1]
            sched_str += "U(L" + str(dim_index) + "," + schedule["unrolling_factor"] + ")"
            transf_loop_nest[dim_index : dim_index + 1] = (
                dim_name + "_Uouter",
                dim_name + "_Uinner",
            )
    return sched_str

# Separate a computation vector into 3 parts where the middle part is the transformation vectors
def seperate_vector(
    X: torch.Tensor, num_transformations: int = 4, pad: bool = True, pad_amount: int = 5
) -> torch.Tensor:
    batch_size, _ = X.shape
    first_part = X[:, :33]
    second_part = X[:, 33 : 33 + MAX_TAGS * num_transformations]
    third_part = X[:, 33 + MAX_TAGS * num_transformations :]
    vectors = []
    for i in range(num_transformations):
        vector = second_part[:, MAX_TAGS * i : MAX_TAGS * (i + 1)].reshape(batch_size, 1, -1)
        vectors.append(vector)

    if pad:
        for i in range(pad_amount):
            vector = torch.zeros_like(vector)
            vectors.append(vector)
    return (first_part, torch.cat(vectors[0:], dim=1), third_part)

def tree_indices_to_device(node, train_device):
    node['loop_index'] = node['loop_index'].to(train_device, non_blocking=True)
    if 'computations_indices' in node:
        node['computations_indices'] = node['computations_indices'].to(
            train_device, non_blocking=True)
    for child in node['child_list']:
        tree_indices_to_device(child, train_device)    
