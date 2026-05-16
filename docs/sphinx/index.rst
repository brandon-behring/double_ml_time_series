Double ML Time Series
=====================

Book-grade companion documentation for the Double Machine Learning manuscript.
This is not a production deployment package and the API is not yet public-stable.

The current code supports cross-sectional partially linear DML, temporal PLR DML
with lagged controls, time-series cross-validation helpers, HAC inference, and
synthetic companion examples. Deferred topics such as true dynamic g-estimation,
period-specific treatment effects, causal forests, BLP/policy trees, and automatic
stationarity enforcement are documented as future work unless an executable example
or tested API proves otherwise.

.. tip::

   Current status lives in ``docs/CURRENT_STATUS.md``. Superseded roadmaps and reports
   are archived under ``docs/archive/superseded_2026-05-02/``.

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started/installation
      :link-type: doc

      Installation, setup, and executable quickstarts.

   .. grid-item-card:: User Guide
      :link: user_guide/fwl_to_dml
      :link-type: doc

      Mathematical tutorials with worked examples from FWL to DML and temporal PLR.

   .. grid-item-card:: API Reference
      :link: api/dml
      :link-type: doc

      API documentation for estimators, data helpers, validation, and research utilities.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      Runnable example scripts demonstrating the current public contract.

----

.. _chapter-map:

Book Chapter Map
----------------

.. list-table::
   :header-rows: 1
   :widths: 10 40 30 20

   * - Chapter
     - Title
     - Library Module
     - Key Classes/Functions
   * - 1
     - Potential Outcomes + FWL
     - :mod:`src.dml.fwl`
     - :func:`~src.dml.fwl.fwl_estimate`
   * - 2
     - Neyman Orthogonality + DML
     - :mod:`src.dml.robinson`, :mod:`src.dml.double_ml`
     - :func:`~src.dml.double_ml.double_ml`
   * - 3
     - Validation Framework
     - :mod:`src.validation`
     - :class:`~src.validation.dgp_generator.DGPGenerator`
   * - 4
     - Cross-Sectional Application
     - :mod:`src.data.oj_loader`
     - :class:`~src.data.oj_loader.OJDataLoader`
   * - 5
     - Temporal PLR DML
     - :mod:`src.dml.temporal_plr_dml`
     - :class:`~src.dml.temporal_plr_dml.TemporalPLRDML`
   * - 6
     - Panel DML + Rolling Window
     - :mod:`src.dml.temporal_plr_dml`
     - :class:`~src.dml.temporal_plr_dml.RollingWindowDML`, :class:`~src.dml.temporal_plr_dml.PanelDML`
   * - 7
     - FRED Integration
     - :mod:`src.data.fred_loader`
     - :class:`~src.data.fred_loader.FREDLoader`
   * - 8
     - Competitor Pricing DGP
     - :mod:`src.validation.insurance_dgp`
     - :func:`~src.validation.insurance_dgp.create_insurance_dgp`
   * - 9
     - Optional Heterogeneity Discussion
     - Optional external EconML discussion
     - Deferred in this repo
   * - 10
     - Research Pipeline Utilities
     - :mod:`src.production`
     - Demo utilities, not deployment guarantees

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/fwl_to_dml
   user_guide/time_series_dml
   user_guide/production_pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/dml
   api/data
   api/validation
   api/production
   api/sensitivity

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
