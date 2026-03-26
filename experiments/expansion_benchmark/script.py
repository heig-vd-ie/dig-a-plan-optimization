from pathlib import Path
import pandapower as pp
import polars as pl
import json

from data_model.benchmark_expansion import BenchmarkExpansion
from helpers.congestion import (
    check_line_loading,
    check_trafo_loading,
    check_voltage_limits,
    apply_profile_scenario_to_pandapower,
)
from tqdm import tqdm
from helpers import generate_log

log = generate_log(name=__name__)

with open(Path(__file__).parent / "00-settings.json", "r", encoding="utf-8") as f:
    konfig = json.load(f)
benchmark_expansion = BenchmarkExpansion(**konfig)

net0 = pp.from_pickle(Path(__file__).parents[2] / benchmark_expansion.grid.pp_file)
map_df = pl.read_csv(
    Path(__file__).parents[2] / benchmark_expansion.grid.egid_id_mapping_file
).select([pl.col("index"), pl.col("egid").cast(pl.Utf8)])
load_profile_dir = (
    Path(__file__).parents[2] / benchmark_expansion.profiles.load_profiles[0]
)
pv_profile_dir = Path(__file__).parents[2] / benchmark_expansion.profiles.pv_profile
limit_percent = benchmark_expansion.congestion_settings.threshold

net0.sgen = net0.load
net0.sgen["p_mw"] = 0
net0.sgen["q_mvar"] = 0

for year in benchmark_expansion.congestion_settings.years:
    log.info(f"Power flow for year {year}")
    load_parquet_file = (
        load_profile_dir
        / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
    )
    pv_parquet_file = (
        pv_profile_dir
        / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
    )
    load_profiles_all_egids = pl.read_parquet(load_parquet_file).join(
        map_df, on="egid", how="left"
    )
    pv_profiles_all_egids = pl.read_parquet(pv_parquet_file).join(
        map_df, on="egid", how="left"
    )
    time_cols = [c for c in load_profiles_all_egids.columns if c.startswith("_")]

    for t in tqdm(time_cols, desc="Power Flow for periods"):
        
        net_case = apply_profile_scenario_to_pandapower(
            net0=net0,
            load_df=load_profiles_all_egids,
            pv_df=pv_profiles_all_egids,
            tcol=t,
            cosphi=benchmark_expansion.grid.cosφ,
        )
        pp.runpp(net_case)

        cong_lines = check_line_loading(net_case, limit_percent=limit_percent)
        cong_trafos = check_trafo_loading(net_case, limit_percent=limit_percent)
        bus_ou = check_voltage_limits(net_case)

#         n_lines_total = len(net_case.line)
#         n_trafos_total = len(net_case.trafo)

#         print(f"\nYear {year} | Timestamp {tcol}")
#         print(f"Congestion threshold = {LIMIT:.1f}%")

#         print("\nTop congested lines before reinforcement:")
#         print(cong_lines_before[["line_idx", "loading_percent"]].head(10))

#         print("\nTop congested trafos before reinforcement:")
#         print(cong_trafos_before[["trafo_idx", "loading_percent"]].head(10))

#         reinforced_lines = set()
#         reinforced_trafos = set()

#         rounds_used = 0

#         while len(cong_lines) > 0 or len(cong_trafos) > 0:
#             rounds_used += 1

#             net_case.line["loading_percent"] = net_case.res_line["loading_percent"]
#             overloaded_line_idx = net_case.line.index[
#                 net_case.line["loading_percent"] >= LIMIT
#             ].tolist()
#             reinforced_lines.update(map(int, overloaded_line_idx))

#             net_case.line["max_i_ka"] = net_case.line.apply(
#                 lambda x: (
#                     x["max_i_ka"]
#                     if x["loading_percent"] < LIMIT
#                     else x["max_i_ka"] * x["loading_percent"] * 1.2 / 100
#                 ),
#                 axis=1,
#             )
#             net_case.line = net_case.line.drop(columns="loading_percent")

#             net_case.trafo["loading_percent"] = net_case.res_trafo["loading_percent"]

#             overloaded_trafo_idx = net_case.trafo.index[
#                 net_case.trafo["loading_percent"] >= LIMIT
#             ].tolist()
#             reinforced_trafos.update(map(int, overloaded_trafo_idx))

#             net_case.trafo["sn_mva"] = net_case.trafo.apply(
#                 lambda x: (
#                     x["sn_mva"]
#                     if x["loading_percent"] < LIMIT
#                     else x["sn_mva"] * x["loading_percent"] * 1.2 / 100
#                 ),
#                 axis=1,
#             )
#             net_case.trafo = net_case.trafo.drop(columns="loading_percent")

#             pp.runpp(net_case)

#             cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
#             cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

#             if rounds_used >= MAX_ROUNDS:
#                 print(f"Stop reached at={tcol}, of year={year}")
#                 break

#         # Final state at this timestamp
#         final_n_cong_lines = len(cong_lines)
#         final_n_cong_trafos = len(cong_trafos)

#         results.append(
#             {
#                 "year": year,
#                 "time_col": tcol,
#                 "rounds_used": rounds_used,
#                 "n_congested_lines_before": len(cong_lines_before),
#                 "n_congested_trafos_before": len(cong_trafos_before),
#                 "cong_rate_lines_before": (
#                     len(cong_lines_before) / n_lines_total if n_lines_total else 0.0
#                 ),
#                 "cong_rate_trafos_before": (
#                     len(cong_trafos_before) / n_trafos_total if n_trafos_total else 0.0
#                 ),
#                 "final_n_congested_lines": final_n_cong_lines,
#                 "final_n_congested_trafos": final_n_cong_trafos,
#                 "final_cong_rate_lines": (
#                     final_n_cong_lines / n_lines_total if n_lines_total else 0.0
#                 ),
#                 "final_cong_rate_trafos": (
#                     final_n_cong_trafos / n_trafos_total if n_trafos_total else 0.0
#                 ),
#                 "reinforced_lines": ",".join(map(str, sorted(reinforced_lines))),
#                 "reinforced_trafos": ",".join(map(str, sorted(reinforced_trafos))),
#             }
#         )

#         # use the new reinforced capacities as the start for the next time
#         net_plan = copy.deepcopy(net_case)

#         line_snapshot = net_plan.line["max_i_ka"].copy()
#         line_snapshot.name = tcol
#         capacity_history_lines.append(line_snapshot)

#         trafo_snapshot = net_plan.trafo["sn_mva"].copy()
#         trafo_snapshot.name = tcol
#         capacity_history_trafos.append(trafo_snapshot)

#     # calculation of yearly reinforcement costs
#     delta_i_ka = net_plan.line["max_i_ka"].sub(line_max_i_init, fill_value=0.0)
#     delta_sn_mva = net_plan.trafo["sn_mva"].sub(trafo_sn_init, fill_value=0.0)

#     line_capacity_increase_percent = (
#         (net_plan.line["max_i_ka"] - line_max_i_init)
#         / line_max_i_init.replace(0.0, np.nan)
#         * 100.0
#     )

#     trafo_capacity_increase_percent = (
#         (net_plan.trafo["sn_mva"] - trafo_sn_init)
#         / trafo_sn_init.replace(0.0, np.nan)
#         * 100.0
#     )

#     print(f"\nYear {year} - line capacity increase percent summary:")
#     print(line_capacity_increase_percent.describe())

#     print(f"\nYear {year} - trafo capacity increase percent summary:")
#     print(trafo_capacity_increase_percent.describe())
#     line_cap_df = pd.DataFrame(
#         {
#             "line_idx": net_plan.line.index,
#             "max_i_ka_start": line_max_i_init.values,
#             "max_i_ka_final": net_plan.line["max_i_ka"].values,
#             "capacity_increase_percent": line_capacity_increase_percent.values,
#         }
#     ).sort_values("capacity_increase_percent", ascending=False)

#     trafo_cap_df = pd.DataFrame(
#         {
#             "trafo_idx": net_plan.trafo.index,
#             "sn_mva_start": trafo_sn_init.values,
#             "sn_mva_final": net_plan.trafo["sn_mva"].values,
#             "capacity_increase_percent": trafo_capacity_increase_percent.values,
#         }
#     ).sort_values("capacity_increase_percent", ascending=False)

#     print(f"\nTop 10 reinforced lines in year {year}:")
#     print(line_cap_df.head(10))

#     print(f"\nTop 10 reinforced trafos in year {year}:")
#     print(trafo_cap_df.head(10))

#     lengths_km = net_plan.line.get("length_km", line_length_km).fillna(1.0)
#     from_bus_v_kv = net_plan.line["from_bus"].map(net_plan.bus["vn_kv"]).fillna(0.0)

#     delta_line_kva = np.sqrt(3.0) * from_bus_v_kv * delta_i_ka * 1000.0
#     delta_line_kw = delta_line_kva * grid.cosφ
#     cost_lines = float((delta_line_kw * lengths_km * LINE_COST_PER_KM_KW).sum())

#     delta_trafo_kva = delta_sn_mva * 1000.0
#     delta_trafo_kw = delta_trafo_kva * grid.cosφ
#     cost_trafos = float((delta_trafo_kw * TRAFO_COST_PER_KW).sum())

#     cost_total_year = cost_lines + cost_trafos
#     years_from_base = year - base_year
#     discount_factor = (1.0 + DISCOUNT_RATE) ** years_from_base
#     npv_cost_year = cost_total_year / discount_factor

#     yearly_results.append(
#         {
#             "year": year,
#             "npv_cost_total": npv_cost_year,
#         }
#     )

#     print(
#         f"Year {year} reinforcement cost: CHF {cost_total_year:,.2f} | NPV Cost: CHF {npv_cost_year:,.2f}"
#     )


# # %% Final summary
# summary_df = pd.DataFrame(results)
# yearly_df = pd.DataFrame(yearly_results)
# display(summary_df)  # type: ignore
# display(yearly_df)  # type: ignore

# total_npv_chf = yearly_df["npv_cost_total"].sum()
# total_cost_mchf = total_npv_chf / 1e6

# print(f"Total cost [MCHF]: {total_cost_mchf:.6f}")

# # %% plotting the evolution of total installed capacity over time
# line_hist_df = pd.DataFrame(capacity_history_lines)
# trafo_hist_df = pd.DataFrame(capacity_history_trafos)

# line_hist_df.index.name = "time_col"
# trafo_hist_df.index.name = "time_col"

# # Total installed capacity evolution
# line_total = line_hist_df.sum(axis=1)
# trafo_total = trafo_hist_df.sum(axis=1)

# x_line = range(len(line_total))
# x_trafo = range(len(trafo_total))

# plt.figure(figsize=(10, 5))
# plt.plot(x_line, line_total.values)
# plt.xlabel("Hour index in 2025")
# plt.ylabel("Total line max_i_ka")
# plt.title("Evolution of total line capacity in 2025")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(x_trafo, trafo_total.values)
# plt.xlabel("Hour index in 2025")
# plt.ylabel("Total transformer sn_mva")
# plt.title("Evolution of total transformer capacity in 2025")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # %%

# # %%
