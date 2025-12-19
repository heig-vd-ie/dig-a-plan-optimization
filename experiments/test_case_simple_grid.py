# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from experiments import *


# %%
def create_simple_grid():
    net = create_cigre_network_mv(with_der="all")  # type: ignore

    bus: pl.DataFrame = pl.from_pandas(net["bus"])
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

    trafo: pl.DataFrame = pl.from_pandas(net["trafo"])

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
        )
        .with_row_index(name="bus", offset=bus.height)
        .with_columns(
            ("switch " + c("name").cast(pl.Utf8)).alias("name"),
        )
    )
    new_bus_mapping = pl_to_dict(switch["line_name", "bus"])

    line = line.with_columns(
        c("name").replace_strict(new_bus_mapping, default=None).alias("to_bus"),
    )

    bus = pl.concat(
        [
            bus,
            switch.select(
                ("Bus " + c("bus").cast(pl.Utf8)).alias("name"),
                pl.lit(20).alias("vn_kv"),
                pl.lit("b").alias("type"),
                pl.lit("CIGRE_MV").alias("zone"),
                pl.lit(True).alias("in_service"),
            ),
        ],
        how="diagonal_relaxed",
    )

    net["bus"] = bus.to_pandas()
    net["switch"] = switch.drop("line_name").to_pandas()
    net["line"] = line.to_pandas()
    net["trafo"] = trafo.to_pandas()
    net["storage"] = net["storage"].iloc[:0]

    net["load"] = net["load"][net["load"]["p_mw"] < 2]

    net["trafo"]["shift_degree"] = 0
    net["line"].loc[:, "max_i_ka"] = 1
    pp.to_pickle(net, "examples/ieee-33/simple_grid.p")


# %%
create_simple_grid()
