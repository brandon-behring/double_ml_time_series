Installation
============

Requirements
------------

- **Python 3.11+** (required)
- NumPy, SciPy, pandas, scikit-learn, statsmodels (core dependencies)
- Optional: FRED API key for macroeconomic data access

Install from Source
-------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/bbehring/double_ml_time_series.git
   cd double_ml_time_series
   pip install -e .
   pip install -r requirements.txt

Verify Installation
-------------------

.. code-block:: python

   # Core estimators
   from src.dml import double_ml, DynamicDML, RollingWindowDML
   print("DML estimators OK")

   # Data loaders
   from src.data import FREDLoader, OJDataLoader
   print("Data loaders OK")

   # Validation
   from src.validation import DGPGenerator, TimeSeriesDGPGenerator
   print("Validation OK")

Optional: FRED API Key
-----------------------

To use the :class:`~src.data.fred_loader.FREDLoader` with live FRED data,
obtain a free API key from `FRED <https://fred.stlouisfed.org/docs/api/api_key.html>`_:

.. code-block:: bash

   export FRED_API_KEY="your_key_here"

The library provides :func:`~src.data.fred_loader.create_synthetic_fred_data`
for offline development and testing without an API key.

Building Documentation
-----------------------

.. code-block:: bash

   pip install -r requirements-docs.txt
   cd docs/sphinx
   make html
   # Open _build/html/index.html
