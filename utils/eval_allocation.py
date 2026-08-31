def effective_num_envs(configured_num_envs: int, eval_num: int) -> int:
    """Do not allocate vector environments that cannot receive an episode."""
    if any(type(value) is not int or value < 1 for value in (configured_num_envs, eval_num)):
        raise ValueError("num_envs and eval_num must be positive integers")
    return min(configured_num_envs, eval_num)
