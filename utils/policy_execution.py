"""Bound inference chunks before passing them to the unchanged policy eval loop."""

import os


def validate_horizon(policy_name, horizon):
    if policy_name == "Pi_05":
        if type(horizon) is not int or horizon < 1:
            raise ValueError("Pi_05 execution_horizon must be a positive integer")
    elif horizon is not None:
        raise ValueError("execution_horizon is only supported for Pi_05")


def execution_client(policy_name, client):
    if policy_name != "Pi_05":
        return client
    raw = os.environ.get("PI05_EXECUTION_HORIZON")
    if raw is None and not os.environ.get("SIM_SERVICE_SESSION_ID"):
        return client  # Unconfigured native evaluation keeps upstream behavior.
    if raw is None or not raw.isascii() or not raw.isdecimal():
        raise ValueError("PI05_EXECUTION_HORIZON must be a positive integer")
    horizon = int(raw)
    validate_horizon(policy_name, horizon)
    return _ExecutionClient(client, horizon)


def _chunk_length(chunk):
    # Policy transport returns indexable action sequences (including ndarrays).
    # Reject mappings/strings and empty results before the upstream loop runs.
    if isinstance(chunk, (dict, str, bytes)):
        raise ValueError("Policy returned an invalid action chunk")
    try:
        length = len(chunk)
        chunk[:0]
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError("Policy returned an invalid action chunk") from error
    if length == 0:
        raise ValueError("Policy returned an empty action chunk")
    return length


class _ExecutionClient:
    def __init__(self, client, horizon):
        self.client = client
        self.horizon = horizon

    def call(self, func_name, **kwargs):
        result = self.client.call(func_name=func_name, **kwargs)
        if func_name == "get_action":
            _chunk_length(result)
            return result[:self.horizon]
        if func_name == "get_action_batch":
            batch_size = _chunk_length(result)
            indices = kwargs.get("env_idx_list", kwargs.get("obs"))
            if indices is not None and batch_size != len(indices):
                raise ValueError("Policy batch size does not match active environments")
            lengths = [_chunk_length(chunk) for chunk in result]
            if len(set(lengths)) != 1:
                raise ValueError("Policy batch action chunks have different lengths")
            return [chunk[:self.horizon] for chunk in result]
        return result

    def __getattr__(self, name):
        return getattr(self.client, name)
