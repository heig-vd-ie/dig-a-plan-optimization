import pandas as pd


def create_visualization_data(collections_dict, risk_info_dict):
    cursors = {
        name: collection.find({}, {"objectives": 1, "_source_file": 1, "_id": 0})
        for name, collection in collections_dict.items()
    }

    data = {}
    dfs = {}

    for name, cursor in cursors.items():
        data[name] = []
        risk_method = risk_info_dict[name]["method"]
        risk_param = risk_info_dict[name]["parameter"]

        for doc in cursor:
            file_name = doc.get("_source_file", "unknown")
            for obj in doc.get("objectives", []):
                data[name].append(
                    {
                        "objective": obj,
                        "iteration": int(file_name.split("_")[-1].split(".")[0]),
                        "risk_method": risk_method,
                        "risk_parameter": risk_param,
                        "risk_label": f"{risk_method}"
                        + (f" ({risk_param})" if risk_param is not None else ""),
                    }
                )

        dfs[name] = pd.DataFrame(data[name])
        dfs[name]["source"] = name

    df_all = pd.concat(dfs.values(), ignore_index=True)

    return df_all[
        (df_all["risk_method"] != "Expectation")
        | (df_all["iteration"] == df_all["iteration"].max())
    ]


def create_summary_stats(df):
    summary = (
        df.groupby(["risk_method", "risk_parameter"])["objective"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )
    return summary


def filter_data_by_risk_method(df, risk_methods):
    if isinstance(risk_methods, str):
        risk_methods = [risk_methods]
    return df[df["risk_method"].isin(risk_methods)]


def filter_data_by_iteration_range(df, min_iter=None, max_iter=None):
    if min_iter is not None:
        df = df[df["iteration"] >= min_iter]
    if max_iter is not None:
        df = df[df["iteration"] <= max_iter]
    return df
