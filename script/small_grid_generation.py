from pandapower.networks import create_cigre_network_mv

import pandapower as pp
import polars as pl
from polars import col as c
from shapely import  from_geojson
from general_function import pl_to_dict, build_non_existing_dirs


if __name__ == "__main__":
    net = create_cigre_network_mv(with_der="all") # type: ignore

    bus: pl.DataFrame = pl.from_pandas(net["bus"]).with_columns(
        c("geo").map_elements(lambda x: list(from_geojson(x).coords)[0], return_dtype=pl.List(pl.Float64)).alias("coords"),
    ).with_row_index(name="bus_id")

    geo_mapping = pl_to_dict(bus["bus_id", "coords"])

    line: pl.DataFrame = pl.from_pandas(net["line"])

    line = pl.concat([
        line,
        line.slice(-2).with_columns(
            pl.Series(["Line 3-10", "Line 11-13"]).alias("name"),
            pl.Series([6, 11]).alias("from_bus"),
            pl.Series([10, 13]).alias("to_bus")
        )
    ], how="diagonal_relaxed")


    line = line.with_columns(
            pl.concat_list(c("from_bus", "to_bus").replace_strict(geo_mapping, default=None)).alias("coords"),
        ).with_columns(
            c("coords").list.gather_every(n = 2).alias("x_coords"),
            c("coords").list.gather_every(n = 2, offset=1).alias("y_coords"),
        )
        
    trafo: pl.DataFrame = pl.from_pandas(net["trafo"])

    trafo = trafo.with_columns(
            pl.concat_list(c("hv_bus", "lv_bus").replace_strict(geo_mapping, default=None)).alias("coords"),
        ).with_columns(
            c("coords").list.gather_every(n = 2).alias("x_coords"),
            c("coords").list.gather_every(n = 2, offset=1).alias("y_coords"),
        )
        
    switch = line.rename({"name": "line_name"}).with_row_index(name="name")\
    .select(
        c("line_name"),
        c("to_bus").alias("element"),
        pl.lit("b").alias("et"),
        pl.lit("LBS").alias("type"),
        pl.lit(True).alias("closed"),
        c("name"),
        pl.lit(0.0).alias("z_ohm"),
        pl.lit(100.0).alias("in_ka"),
        pl.concat_list(c("x_coords").list.mean(), c("x_coords").list.get(1)).alias("x_coords"),
        pl.concat_list(c("y_coords").list.mean(), c("y_coords").list.get(1)).alias("y_coords"),
    ).with_row_index(name="bus", offset=bus.height)\
    .with_columns(
        ("switch " + c("name").cast(pl.Utf8)).alias("name"),
    )
    new_bus_mapping = pl_to_dict(switch["line_name", "bus"])

    line = line.with_columns(
        pl.concat_list(c("x_coords").list.get(0), c("x_coords").list.mean()).alias("x_coords"),
        pl.concat_list(c("y_coords").list.get(0), c("y_coords").list.mean()).alias("y_coords"),
        c("name").replace_strict(new_bus_mapping, default=None).alias("to_bus")
    )

    bus = pl.concat([
        bus,
        switch.select(
        c("bus").alias("bus_id"),
        ("Bus " + c("bus").cast(pl.Utf8)).alias("name"),
        pl.lit(20).alias("vn_kv"),
        pl.lit("b").alias("type"),
        pl.lit("CIGRE_MV").alias("zone"),
        pl.lit(True).alias("in_service"),
        pl.concat_list(c("x_coords").list.get(0), c("y_coords").list.get(0)).alias("coords"),
    )
    ], how="diagonal_relaxed")

    net["bus"] = bus.drop("geo").to_pandas()
    net["switch"] = switch.drop("line_name").to_pandas()
    net["line"] = line.drop(["geo", "coords"]).to_pandas()
    net["trafo"] = trafo.drop(["coords"]).to_pandas()
    net["storage"] = net["storage"].iloc[:0]

    net["load"] = net["load"][net["load"]["p_mw"] < 2]

    net["trafo"]["shift_degree"] = 0


    build_non_existing_dirs(".cache/input_data")
    pp.to_pickle(net, ".cache/input_data/mv_example.p")
