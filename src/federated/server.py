"""
Federated server — FedAvg aggregation.

Averages model weights from all clients, weighted by sample count.
"""

import copy
import torch


def federated_averaging(client_states, client_sizes):
    """
    FedAvg: weighted average of state_dicts by sample count.

    client_states: list of state_dicts
    client_sizes:  list of ints (num samples per client)
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    # Start with first client's state
    avg_state = copy.deepcopy(client_states[0])

    # Zero out all tensors
    for k in avg_state:
        if avg_state[k].dtype.is_floating_point:
            avg_state[k] = avg_state[k] * weights[0]
        else:
            # Non-float buffers (like num_batches_tracked) — take first
            avg_state[k] = client_states[0][k].clone()

    # Add weighted contributions
    for i in range(1, len(client_states)):
        for k in avg_state:
            if avg_state[k].dtype.is_floating_point:
                avg_state[k] += client_states[i][k] * weights[i]

    return avg_state
