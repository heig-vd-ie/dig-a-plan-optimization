
import polars as pl
from polars import col as c
import pyomo.environ as pyo


def extract_optimization_results(model_instance: pyo.Model, var_name: str) -> pl.DataFrame:
    
    index_list: list[str] = list(map(lambda x: x.name, getattr(model_instance, var_name).index_set().subsets()))
        
    if len(index_list) == 1:
        data_pl: pl.DataFrame = pl.DataFrame(
            map(list, getattr(model_instance, var_name).extract_values().items()), 
            schema= [index_list[0], var_name]
        )
    else:
        data_pl: pl.DataFrame = pl.DataFrame(
            map(list, getattr(model_instance, var_name).extract_values().items()), 
            schema= ["index", var_name]
        ).with_columns(
            c("index").list.to_struct(fields=index_list)
        ).unnest("index")
    return data_pl.with_columns(c(index_list))