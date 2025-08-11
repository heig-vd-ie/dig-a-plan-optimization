"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
from pathlib import Path


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


@pytest.fixture
def default_scenario_params():
    """Default parameters for scenario generation."""
    return {
        "δ_load_var": 0.1,
        "δ_pv_var": 0.1,
        "δ_b_var": 0.1,
        "number_of_scenarios": 100,
        "number_of_stages": 3,
        "seed_number": 42,
    }


@pytest.fixture
def default_planning_params():
    """Default planning parameters."""
    return {
        "n_stages": 3,
        "initial_budget": 100000,
        "discount_rate": 0.05,
    }


@pytest.fixture
def default_additional_params():
    """Default additional parameters."""
    return {
        "iteration_limit": 10,
        "n_simulations": 20,
        "risk_measure_param": 0.1,
        "seed": 42,
    }
