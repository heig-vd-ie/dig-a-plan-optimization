from data_exporter.kace_to_dap import kace4reconfiguration
from data_model.reconfiguration import CombinedInput, ReconfigurationOutput
from pipelines.reconfiguration.configs import CombinedConfig, PipelineType
from pipelines.reconfiguration import DigAPlanCombined


def run_combined(request: CombinedInput) -> ReconfigurationOutput:
    base_grid_data = kace4reconfiguration(
        grid=request.grid,
        load_profiles=request.load_profiles,
        st_scenarios=request.scenarios,
        seed=request.seed,
    )
    config = CombinedConfig(
        verbose=True,
        big_m=1e3,
        ε=0.1,
        pipeline_type=PipelineType.COMBINED,
        γ_infeasibility=1.0,
        γ_admm_penalty=0.0,
    )
    dig_a_plan = DigAPlanCombined(config=config)
    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model(groups=request.groups)  # one‐shot solve
    switches = dig_a_plan.result_manager.extract_switch_status()
    voltages = dig_a_plan.result_manager.extract_node_voltage()
    currents = dig_a_plan.result_manager.extract_edge_current()
    taps = dig_a_plan.result_manager.extract_transformer_tap_position()
    return ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )

    return result
