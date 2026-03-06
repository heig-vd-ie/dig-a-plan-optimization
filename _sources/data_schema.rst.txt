Data schema
=================================

To run the DistFlow algorithm, a data schema have been defined. This schema will encompass the 
complete grid topology required for algorithm execution. The Distflow class is populated with data through two 
tables: one representing edges (branch, switch, and transformer equipment) and the other representing nodes. If the 
data does not conform to the expected format, the class will return an error, and the data will not be loaded.

Node Data
-------------------------------------

.. csv-table::
    :file: /tables/node_data_docs.csv
    :header-rows: 1

Edge Data
-------------------------------------

.. csv-table::
    :file: /tables/edge_data_docs.csv
    :header-rows: 1

