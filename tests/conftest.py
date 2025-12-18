"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import pandapower as pp
from pathlib import Path
from pipelines.helpers.json_rw import load_obj_from_json
from data_model.reconfiguration import BenderInput
from pipelines.reconfiguration.configs import (
    ADMMConfig,
    CombinedConfig,
    BenderConfig,
)


@pytest.fixture(scope="session")
def test_cache_dir():
    """Provide a test cache directory that's cleaned up after tests."""
    cache_dir = Path(".cache/test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir
    # Cleanup could be added here if needed


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path("examples")


@pytest.fixture(scope="session")
def test_simple_grid():
    """Provide a simple grid test case."""
    return pp.from_pickle("examples/ieee-33/simple_grid.p")


@pytest.fixture(scope="session")
def test_simple_grid_groups():
    """Provide groups for the simple grid test case."""
    return {
        0: [19, 20, 21, 29, 32, 35],
        1: [35, 30, 33, 25, 26, 27],
        2: [27, 32, 22, 23, 34],
        3: [31, 24, 28, 21, 22, 23],
        4: [34, 26, 25, 24, 31],
    }


@pytest.fixture(scope="session")
def test_bender_config() -> BenderConfig:
    """Provide Bender configuration for the test case."""
    return BenderConfig(
        verbose=False,
        threads=1,
        big_m=1e2,
        factor_p=1e-3,
        factor_q=1e-3,
        factor_v=1,
        factor_i=1e-3,
        master_relaxed=False,
    )


@pytest.fixture(scope="session")
def test_combined_config() -> CombinedConfig:
    return CombinedConfig(
        verbose=False,
        threads=1,
        big_m=1e2,
        γ_infeasibility=1.0,
        factor_v=1,
        factor_i=1e-3,
    )


@pytest.fixture(scope="session")
def test_admm_config() -> ADMMConfig:
    return ADMMConfig(
        verbose=False,
        threads=1,
        solver_name="gurobi",
        solver_non_convex=2,
        big_m=1e3,
        ε=1e-4,
        ρ=2.0,  # initial rho
        γ_infeasibility=1.0,
        γ_admm_penalty=1.0,
        max_iters=10,
        μ=10.0,
        τ_incr=2.0,
        τ_decr=2.0,
        groups=1,
    )


@pytest.fixture(scope="session")
def bender_input_payload() -> BenderInput:
    return BenderInput(
        **load_obj_from_json(Path("examples/payloads/reconfiguration/ex1-bender.json"))
    )


@pytest.fixture(scope="session")
def test_basic_grid_quick_expansion() -> Path:
    return Path("examples/payloads/simple_grid_quick_test.json")
