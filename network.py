# netowrk.py
import numpy as np
import copy
import torch

def set_network_topology(topology: str, num_agents: int, random_p_edge: float = 0.5) -> np.ndarray:
    """
    Creates a binary adjacency matrix for a given network topology.

    The adjacency matrix `adj_matrix` is a square matrix of size (num_agents, num_agents)
    where `adj_matrix[i, j] = 1` if agent i and agent j can communicate, and 0 otherwise.
    The matrix is symmetric and includes self-loops (`adj_matrix[i, i] = 1`), as each
    agent should consider its own model during the averaging step.

    Args:
        topology (str): The type of network topology ('ring', 'fully_connected', 'random').
        num_agents (int): The number of agents in the decentralized learning setup.
        random_p_edge (float): The probability of an edge existing between any two
                               nodes in a 'random' topology. Must be high enough
                               to ensure a connected graph can be generated.

    Returns:
        np.ndarray: The (num_agents, num_agents) binary adjacency matrix.
    """
    print(f"Setting up a '{topology}' network topology for {num_agents} agents...")
    adj_matrix = np.zeros((num_agents, num_agents), dtype=int)

    if topology == 'ring':
        # Each agent is connected to itself and its two immediate neighbors
        for i in range(num_agents):
            adj_matrix[i, i] = 1
            adj_matrix[i, (i - 1) % num_agents] = 1
            adj_matrix[i, (i + 1) % num_agents] = 1

    elif topology == 'fully_connected':
        # Every agent is connected to every other agent
        adj_matrix = np.ones((num_agents, num_agents), dtype=int)

    elif topology == 'random':
        # Generate a random graph and ensure it is connected.
        is_connected = False
        while not is_connected:
            # Generate a random symmetric matrix based on edge probability
            adj_matrix = np.triu(np.random.binomial(1, random_p_edge, (num_agents, num_agents)), 1)
            adj_matrix += adj_matrix.T
            # All agents have a self-loop
            np.fill_diagonal(adj_matrix, 1)

            # Check for connectivity using the graph's Laplacian matrix.
            # A graph is connected if and only if the second smallest eigenvalue
            # of its Laplacian is greater than 0.
            degrees = np.sum(adj_matrix, axis=1)
            laplacian = np.diag(degrees) - adj_matrix
            eigenvalues = np.linalg.eigvalsh(laplacian)

            # The smallest eigenvalue is always 0. We check the next one.
            second_smallest_eigenvalue = eigenvalues[1]

            if second_smallest_eigenvalue > 1e-10:  # Check with a small tolerance
                is_connected = True
            else:
                print("Generated random graph is not connected, retrying...")

    else:
        raise ValueError(f"Unknown network topology: {topology}. Please use 'ring', 'fully_connected', or 'random'.")

    print("Network topology successfully created.")
    return adj_matrix


# Diffusion function for gossip averaging
def gossip_average(agent_models: list, adj_matrix: np.ndarray):
    """
    Performs one round of gossip averaging based on the network topology.
    Each agent averages its model with those of its neighbors.
    """
    num_agents = len(agent_models)
    old_states = [copy.deepcopy(model.state_dict()) for model in agent_models]

    for i in range(num_agents):
        # Find the neighbors of agent i from the adjacency matrix
        neighbor_indices = [j for j, connected in enumerate(adj_matrix[i]) if connected]

        # Start with its own model's state
        new_state_dict = copy.deepcopy(old_states[i])

        # Add the states from its neighbors (excluding itself, as it's already included)
        for neighbor_idx in neighbor_indices:
            if neighbor_idx != i:
                for key in new_state_dict.keys():
                    new_state_dict[key] += old_states[neighbor_idx][key]

        # Average by the number of models (self + neighbors)
        num_connections = len(neighbor_indices)
        for key in new_state_dict.keys():
            new_state_dict[key] = torch.div(new_state_dict[key], num_connections)

        # Load the new averaged state into the agent's model
        agent_models[i].load_state_dict(new_state_dict)

    return agent_models
