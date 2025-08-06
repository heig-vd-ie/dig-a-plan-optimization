# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from examples import *


# %%
def create_simple_grid():
    net = create_cigre_network_mv(with_der="all")  # type: ignore

    bus: pl.DataFrame = pl.from_pandas(net["bus"])
    bus: pl.DataFrame = bus.with_columns(
        c("geo")
        .map_elements(
            lambda x: list(from_geojson(x).coords)[0],
            return_dtype=pl.List(pl.Float64),
        )
        .alias("coords"),
    ).with_row_index(name="bus_id")

    geo_mapping = pl_to_dict(bus["bus_id", "coords"])

    line: pl.DataFrame = pl.from_pandas(net["line"])

    line = pl.concat(
        [
            line,
            line.slice(-2).with_columns(
                pl.Series(["Line 3-10", "Line 11-13"]).alias("name"),
                pl.Series([6, 11]).alias("from_bus"),
                pl.Series([10, 13]).alias("to_bus"),
            ),
        ],
        how="diagonal_relaxed",
    )

    line = line.with_columns(
        pl.concat_list(
            c("from_bus", "to_bus").replace_strict(geo_mapping, default=None)
        ).alias("coords"),
    ).with_columns(
        c("coords").list.gather_every(n=2).alias("x_coords"),
        c("coords").list.gather_every(n=2, offset=1).alias("y_coords"),
    )

    trafo: pl.DataFrame = pl.from_pandas(net["trafo"])

    trafo = trafo.with_columns(
        pl.concat_list(
            c("hv_bus", "lv_bus").replace_strict(geo_mapping, default=None)
        ).alias("coords"),
    ).with_columns(
        c("coords").list.gather_every(n=2).alias("x_coords"),
        c("coords").list.gather_every(n=2, offset=1).alias("y_coords"),
    )

    switch = (
        line.rename({"name": "line_name"})
        .with_row_index(name="name")
        .select(
            c("line_name"),
            c("to_bus").alias("element"),
            pl.lit("b").alias("et"),
            pl.lit("LBS").alias("type"),
            pl.lit(True).alias("closed"),
            c("name"),
            pl.lit(0.0).alias("z_ohm"),
            pl.lit(100.0).alias("in_ka"),
            pl.concat_list(c("x_coords").list.mean(), c("x_coords").list.get(1)).alias(
                "x_coords"
            ),
            pl.concat_list(c("y_coords").list.mean(), c("y_coords").list.get(1)).alias(
                "y_coords"
            ),
        )
        .with_row_index(name="bus", offset=bus.height)
        .with_columns(
            ("switch " + c("name").cast(pl.Utf8)).alias("name"),
        )
    )
    new_bus_mapping = pl_to_dict(switch["line_name", "bus"])

    line = line.with_columns(
        pl.concat_list(c("x_coords").list.get(0), c("x_coords").list.mean()).alias(
            "x_coords"
        ),
        pl.concat_list(c("y_coords").list.get(0), c("y_coords").list.mean()).alias(
            "y_coords"
        ),
        c("name").replace_strict(new_bus_mapping, default=None).alias("to_bus"),
    )

    bus = pl.concat(
        [
            bus,
            switch.select(
                c("bus").alias("bus_id"),
                ("Bus " + c("bus").cast(pl.Utf8)).alias("name"),
                pl.lit(20).alias("vn_kv"),
                pl.lit("b").alias("type"),
                pl.lit("CIGRE_MV").alias("zone"),
                pl.lit(True).alias("in_service"),
                pl.concat_list(
                    c("x_coords").list.get(0), c("y_coords").list.get(0)
                ).alias("coords"),
            ),
        ],
        how="diagonal_relaxed",
    )

    net["bus"] = bus.drop("geo").to_pandas()
    net["switch"] = switch.drop("line_name").to_pandas()
    net["line"] = line.drop(["geo", "coords"]).to_pandas()
    net["trafo"] = trafo.drop(["coords"]).to_pandas()
    net["storage"] = net["storage"].iloc[:0]

    net["load"] = net["load"][net["load"]["p_mw"] < 2]

    net["trafo"]["shift_degree"] = 0

    LOAD_FACTOR = 1
    TEST_CONFIG = [
        {"line_list": [], "switch_list": []},
        {"line_list": [6, 9], "switch_list": [25, 28]},
        {"line_list": [2, 6, 9], "switch_list": [21, 25, 28]},
        {"line_list": [16], "switch_list": [35]},
        {"line_list": [1], "switch_list": [20]},
        {"line_list": [10], "switch_list": [29]},
        {"line_list": [7, 11], "switch_list": [26, 30]},
    ]
    NB_TEST = 0

    net["load"]["p_mw"] = net["load"]["p_mw"] * LOAD_FACTOR
    net["load"]["q_mvar"] = net["load"]["q_mvar"] * LOAD_FACTOR

    net["line"].loc[:, "max_i_ka"] = 1
    net["line"].loc[TEST_CONFIG[NB_TEST]["line_list"], "max_i_ka"] = 1e-2

    build_non_existing_dirs("data")
    pp.to_pickle(net, "data/simple_grid.p")


# %%
create_simple_grid()
