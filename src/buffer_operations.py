import torch
import numpy as np


def reshuffles_from_bays(bays: torch.tensor):
    assert bays.dim() == 4
    flipped = torch.flip(bays, dims=[2])
    cummin_flipped, _ = torch.cummin(flipped, dim=2)
    min_below = torch.flip(cummin_flipped, dims=[2])
    is_reshuffle = bays > min_below
    return is_reshuffle.float()


def will_result_in_reshuffle(bays: torch.tensor, flat_T: torch.tensor):
    assert bays.dim() == 4 and flat_T.dim() == 2
    N = get_n_from_flat_T(flat_T)
    next_container = get_next_container(flat_T, N)
    remaining_ports = get_remaining_ports(flat_T, N)
    normalized_next_container = next_container / remaining_ports
    normalized_next_container = normalized_next_container.unsqueeze(1).unsqueeze(1)
    min_containers = get_min_containers(bays)
    would_reshuffle = normalized_next_container > min_containers
    return would_reshuffle.squeeze(1).float()


def get_min_containers(bays: torch.tensor):
    mask = bays != 0
    masked_tensor = torch.where(mask, bays, torch.tensor(float("inf")))
    min_values = torch.min(masked_tensor, dim=2)[0]
    return min_values


def get_index_of_last_nonzero(arr: torch.tensor):
    flipped = arr.flip(dims=[1])
    reversed_indices = torch.argmax((flipped != 0).int(), dim=1)
    original_indices = arr.size(1) - 1 - reversed_indices
    return original_indices


def get_next_container(flat_T: torch.tensor, N: int):
    flat_T = flat_T[:, : N - 1]
    return get_index_of_last_nonzero(flat_T) + 1


def get_n_from_flat_T(flat_array: torch.tensor):
    width = flat_array.shape[1]
    N = (1 + np.sqrt(1 + 8 * width)) / 2
    return np.round(N).astype(int)


def get_remaining_ports(flat_array: torch.tensor, N: int):
    T = reshape_to_upper_triangular(flat_array, N)
    T = torch.max(T, dim=1).values
    return get_index_of_last_nonzero(T)


def reshape_to_upper_triangular(flat_array: torch.tensor, n: int):
    matrices = torch.zeros((flat_array.shape[0], n, n), dtype=flat_array.dtype)
    triu_indices = torch.triu_indices(n, n, offset=1)
    matrices[:, triu_indices[0], triu_indices[1]] = flat_array
    return matrices
