Installation
============

Requirements
------------

- Python 3.11+
- NumPy, SciPy, pandas, scikit-learn, statsmodels
- Optional: FRED API key for live macroeconomic data
- Optional docs dependencies for Sphinx builds

Install From Source
-------------------

.. code-block:: bash

   cd double_ml_time_series
   venv/bin/python -m pip install -e ".[dev]"

For documentation work:

.. code-block:: bash

   venv/bin/python -m pip install -e ".[dev,docs]"

Verify Installation
-------------------

.. code-block:: bash

   venv/bin/python -c "from src.dml import double_ml, TemporalPLRDML, RollingWindowDML; print('OK')"
   venv/bin/python -c "from src.data import FREDLoader, create_synthetic_fred_data; print('OK')"
   venv/bin/python -c "from src.validation import create_insurance_dgp; print('OK')"

Optional: FRED API Key
----------------------

Live FRED access requires an API key:

.. code-block:: bash

   export FRED_API_KEY="your_key_here"

Use :func:`~src.data.fred_loader.create_synthetic_fred_data` for offline examples and
tests that should not depend on network access.

Building Documentation
----------------------

.. code-block:: bash

   venv/bin/python -m sphinx -b html -W --keep-going docs/sphinx docs/sphinx/_build/html
