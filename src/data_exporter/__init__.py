import polars as pl
import patito as pt


def validate_data(df: pl.DataFrame, model) -> pt.DataFrame:
    df_pt = (
        pt.DataFrame(df)
        .set_model(model)
        .fill_null(strategy="defaults")
        .cast(strict=True)
    )
    df_pt.validate()
    return df_pt
