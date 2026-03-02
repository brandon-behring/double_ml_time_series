Double ML Time Series
=====================

**Volume 2: Double Machine Learning for Time Series Causal Inference**

A production-grade Python library for Double Machine Learning (DML) with time series
extensions, developed as the companion code to a 180-page academic reference book on
causal inference for insurance and annuity competitor pricing with macroeconomic controls.

.. tip::

   This library accompanies a LaTeX reference book (10 chapters + Julia appendix).
   The :ref:`chapter-map` below maps each book chapter to its corresponding library module.

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started/installation
      :link-type: doc

      Installation, setup, and a 5-minute quickstart.

   .. grid-item-card:: User Guide
      :link: user_guide/fwl_to_dml
      :link-type: doc

      Mathematical tutorials with worked examples — FWL to DML,
      time series extensions, and production deployment.

   .. grid-item-card:: API Reference
      :link: api/dml
      :link-type: doc

      Complete API documentation for all modules — estimators,
      data loaders, validation, production, and sensitivity.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      Runnable example scripts demonstrating key workflows.

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
     - Dynamic Treatment Effects
     - :mod:`src.dml.dynamic_dml`
     - :class:`~src.dml.dynamic_dml.DynamicDML`
   * - 6
     - Panel DML + Rolling Window
     - :mod:`src.dml.dynamic_dml`
     - :class:`~src.dml.dynamic_dml.RollingWindowDML`, :class:`~src.dml.dynamic_dml.PanelDML`
   * - 7
     - FRED Integration
     - :mod:`src.data.fred_loader`
     - :class:`~src.data.fred_loader.FREDLoader`
   * - 8
     - Competitor Pricing
     - :mod:`src.validation.insurance_dgp`
     - :func:`~src.validation.insurance_dgp.create_insurance_dgp`
   * - 9
     - Heterogeneity Analysis
     - :mod:`src.sensitivity`
     - :class:`~src.sensitivity.rosenbaum.RosenbaumBounds`
   * - 10
     - Production Pipeline
     - :mod:`src.production`
     - :class:`~src.production.dml_pipeline.InsuranceDMLPipeline`

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
