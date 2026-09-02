"""Select active camera rows before synchronizing/copying a tensor to CPU."""


def selected_frames_to_numpy(tensor, env_ids):
    selected = tensor[list(env_ids)]
    if hasattr(selected, "cpu"):
        return selected.cpu().numpy()
    return selected
