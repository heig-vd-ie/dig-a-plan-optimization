# TODO: This is a temporary script to record the geolocations of the edges in the networks. It should be removed once we have a better way to store this information in the network files.
# FIXME: This is not a good way to store the geolocations of the edges, as it is not linked to the network files. We should find a better way to store this information in the network files, e.g. by adding a "geo" column to the line and trafo tables in the pandapower network files.
from experiments import *
import plotly.express as px
from shapely.wkt import loads
from copy import deepcopy
import geopandas as gpd
import pandas as pd
import random
from experiments.expansion_planning_script import replace_feeder_in_payload


def random_point_leman_epsg2056():
    E_min, E_max = 2532500, 2537500
    N_min, N_max = 1142500, 1147500

    E = random.uniform(E_min, E_max)
    N = random.uniform(N_min, N_max)

    return E, N


def record_data(kace: str, feedername: str | None = None):
    match kace:
        case "ieee_33":
            payload_file = PROJECT_ROOT / "experiments/ieee_33/00-expansion.json"
        case "boisy":
            payload_file = PROJECT_ROOT / "experiments/boisy/00-expansion.json"
        case "estavayer":
            payload_file = PROJECT_ROOT / "experiments/estavayer/00-expansion.json"
        case _:
            raise ValueError(f"Unknown kace: {kace}")

    with open(payload_file, "r") as f:
        payload = json.load(f)

    if feedername is not None:
        payload = replace_feeder_in_payload(payload, kace, feedername)

    total_load_kw, total_pv_kw = record_load_data(payload)
    gdf = record_geolocation_edges(payload)
    gdf_dict = gdf[["eq_fk", "lat", "lon"]].to_dict(orient="records")
    output_dir = (
        PROJECT_ROOT
        / settings.cache.figures
        / "geolocations"
        / (payload["grid"]["name"] + ".json")
    )
    final_dict = {
        "kace": kace,
        "feedername": feedername,
        "load_kw": total_load_kw,
        "pv_kw": total_pv_kw,
        "geolocations": gdf_dict,
    }
    with open(output_dir, "w") as f:
        json.dump(final_dict, f, indent=2)


def record_load_data(payload: dict):
    total_load_kw = 0
    for load_prof in payload["profiles"]["load_profiles"]:
        df = pl.read_parquet(PROJECT_ROOT / load_prof / "Full_2050.parquet")
        total_load_kw += df.select([pl.all().exclude("egid").sum()]).to_numpy().max()
    total_pv_kw = 0
    for pv_prof in [payload["profiles"]["pv_profile"]]:
        df = pl.read_parquet(PROJECT_ROOT / pv_prof / "Full_2050.parquet")
        total_pv_kw += df.select([pl.all().exclude("egid").sum()]).to_numpy().max()
    return total_load_kw, total_pv_kw


def record_geolocation_edges(
    payload: dict, debug_plot: bool = False
) -> gpd.GeoDataFrame:
    net = pp.from_pickle(PROJECT_ROOT / payload["grid"]["pp_file"])

    if "eq_fk" not in net.line.columns:
        net.line["eq_fk"] = net.line["name"]
    if "eq_fk" not in net.trafo.columns:
        net.trafo["eq_fk"] = net.trafo["name"]
    if "geo" not in net.line.columns:
        net.line["geo"] = None
    if "geo" not in net.trafo.columns:
        net.trafo["geo"] = None

    net.line["geo"] = net.line["geo"].apply(
        lambda x: (
            x
            if pd.notnull(x)
            else "POINT ({} {})".format(*random_point_leman_epsg2056())
        )
    )
    net.trafo["geo"] = net.trafo["geo"].apply(
        lambda x: (
            x
            if pd.notnull(x)
            else "POINT ({} {})".format(*random_point_leman_epsg2056())
        )
    )

    df1 = deepcopy(net.line[["eq_fk", "geo"]])
    df2 = deepcopy(net.trafo[["eq_fk", "geo"]])
    df = pd.concat([df1, df2], ignore_index=True)

    df["geometry"] = df.apply(lambda row: loads(row["geo"]), axis=1)
    df = df.drop(columns=["geo"])
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="epsg:2056")
    gdf = gdf.to_crs(epsg=4326)
    os.makedirs(PROJECT_ROOT / settings.cache.figures / "geolocations", exist_ok=True)

    gdf["lat"] = gdf.geometry.apply(
        lambda geom: geom.centroid.y if geom is not None else None
    )
    gdf["lon"] = gdf.geometry.apply(
        lambda geom: geom.centroid.x if geom is not None else None
    )

    # Plotting for verification
    if debug_plot:
        fig = px.scatter_map(gdf, lat="lat", lon="lon", hover_name="eq_fk", zoom=12)

        fig.update_layout(
            mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        fig.show()
    return gdf[["eq_fk", "lat", "lon"]]


def record_all():
    record_data(kace="ieee_33")
    record_data(kace="boisy", feedername="feeder_1")
    record_data(kace="boisy", feedername="feeder_2")
    record_data(kace="estavayer", feedername="aumont")
    record_data(kace="estavayer", feedername="autoroutes")
    record_data(kace="estavayer", feedername="bel-air")
    record_data(kace="estavayer", feedername="centre_ville")
    record_data(kace="estavayer", feedername="st-aubin")
    record_data(kace="estavayer", feedername="tout_vent")
    record_data(kace="estavayer", feedername="zone_industrielle")


if __name__ == "__main__":
    record_all()
