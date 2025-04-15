import polars as pl

def literal_constraint(field: pl.Expr, values) -> pl.Expr:
    return field.is_in(list(values.__args__)).alias('literal_constraint')


def optional_unique(field: pl.Expr) -> pl.Expr:
    return field.drop_nulls().is_duplicated().sum() == 0


