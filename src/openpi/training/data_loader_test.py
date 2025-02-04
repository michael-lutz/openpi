import dataclasses

import jax
import jax.numpy as jnp

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# helpers for testing multi-JAX-process data loader
_original_process_count = jax.process_count
_original_process_index = jax.process_index

def simulate_process(process_index: int, process_count: int = 2):
    """Override jax.process_count and jax.process_index to simulate a multi-process environment."""
    jax.process_count = lambda: process_count
    jax.process_index = lambda: process_index

def reset_jax_process_functions():
    """Reset jax.process_count and jax.process_index to their original functions."""
    jax.process_count = _original_process_count
    jax.process_index = _original_process_index

def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

def test_torch_data_loader_multi_jax():
    """
    Simulate two JAX processes (process 0 and process 1) and ensure that each
    gets batches with the correct shape and (optionally) different data.
    """
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 32)

    # simulating proc 0
    simulate_process(process_index=0, process_count=2)
    loader0 = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=3)
    batches0 = list(loader0)
    reset_jax_process_functions()

    # simulating proc 1
    simulate_process(process_index=1, process_count=2)
    loader1 = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=3)
    batches1 = list(loader1)
    reset_jax_process_functions()

    assert len(batches0) == len(batches1) and len(batches0) == 3, (
        "Expected 3 batches from each process"
    )

    for batch in batches0:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch)), (
            "Process 0: Incorrect batch size detected"
        )
    for batch in batches1:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch)), (
            "Process 1: Incorrect batch size detected"
        )

    # checking data is different between processes
    # data could be the same during random generation of fake data but will practically never happen
    leaves0 = jax.tree.leaves(batches0[0])
    leaves1 = jax.tree.leaves(batches1[0])
    first_elem0 = leaves0[0][0]
    first_elem1 = leaves1[0][0]
    assert not jnp.allclose(first_elem0, first_elem1), (
        "Expected different data between processes, but found similar values"
    )
