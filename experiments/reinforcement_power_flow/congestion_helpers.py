import pandas as pd
import pandapower as pp


def check_line_loading(
    net: pp.pandapowerNet,
    limit_percent: float | None = None,
) -> pd.DataFrame:
    """
    Returns a line table sorted by pandapower res_line.loading_percent.
    """

    res_cols = [c for c in ["loading_percent", "i_from_ka", "i_to_ka"] if c in net.res_line.columns]

    df = net.line.copy()
    df = df.join(net.res_line[res_cols], how="left")
    df["line_idx"] = df.index

    df = df[df["loading_percent"].notna()]
    df = df.sort_values("loading_percent", ascending=False)

    df = df[df["loading_percent"] >= limit_percent].copy()

    return df


def check_trafo_loading(
    net: pp.pandapowerNet,
    limit_percent: float | None = None,
) -> pd.DataFrame:
    """
    Returns a transformer table sorted by pandapower res_trafo.loading_percent.
    """

    res_cols = [c for c in ["loading_percent"] if c in net.res_trafo.columns]

    df = net.trafo.copy()
    df = df.join(net.res_trafo[res_cols], how="left")
    df["trafo_idx"] = df.index

    df = df[df["loading_percent"].notna()]
    df = df.sort_values("loading_percent", ascending=False)

    df = df[df["loading_percent"] >= limit_percent].copy()

    return df


def reinforce_line_case(
    net: pp.pandapowerNet,
    line_idx: int,
    step_percent: float = 5.0,
) -> pp.pandapowerNet:
    """
    Increase line thermal capacity by increasing max_i_ka.
    """

    net.line.at[line_idx, "max_i_ka"] *= (1.0 + step_percent / 100.0)
    return net

def reinforce_trafo_case(
    net: pp.pandapowerNet,
    trafo_idx: int,
    step_percent: float = 5.0,
) -> pp.pandapowerNet:
    """
    Increase transformer capacity by increasing sn_mva.
    """

    net.trafo.at[trafo_idx, "sn_mva"] *= (1.0 + step_percent / 100.0)
    return net