2. Slave model
=================================


2.1. Sets 
-------------



.. csv-table:: 
   :file: /tables/sets_docs.csv
   :header-rows: 1
   
2.2. Variables
----------------

.. csv-table:: 
   :file: /tables/variable_docs.csv
   :header-rows: 1
   

----------------------

The slack bus (or reference bus) is used to set the overall voltage level of the network. This constraint fixes the squared voltage at the slack node to a predetermined value:

.. automodule:: slave_model.variables
   :no-index:

2.3. Parameters
------------------

.. csv-table:: 
   :file: /tables/parameters_docs.csv
   :header-rows: 1


2.4. Constraints and Objective
----------------------------------

.. automodule:: slave_model.constraints
   :no-index:


