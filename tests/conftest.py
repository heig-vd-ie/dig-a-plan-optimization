"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import pandapower as pp
from pathlib import Path
from pipelines.reconfiguration.configs import (
    ADMMConfig,
    CombinedConfig,
    BenderConfig,
    PipelineType,
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
    return Path("data")


@pytest.fixture(scope="session")
def test_simple_grid():
    """Provide a simple grid test case."""
    return pp.from_pickle("data/simple_grid.p")


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
def test_taps():
    """Provide taps for the simple grid test case."""
    return [95, 98, 99, 100, 101, 102, 105]


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
        pipeline_type=PipelineType.BENDER,
    )


@pytest.fixture(scope="session")
def test_combined_config() -> CombinedConfig:
    return CombinedConfig(
        verbose=False,
        threads=1,
        big_m=1e3,
        γ_infeasibility=1.0,
        factor_p=1e-3,
        factor_q=1e-3,
        factor_v=1,
        factor_i=1e-3,
        pipeline_type=PipelineType.COMBINED,
    )


@pytest.fixture(scope="session")
def test_admm_config() -> ADMMConfig:
    return ADMMConfig(
        verbose=False,
        threads=1,
        pipeline_type=PipelineType.ADMM,
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
