# FRED + Macro Time-Series

Macroeconomic time-series datasets — FRED single-series + bundled databases (FRED-MD, FRED-QD), real-time vintages (ALFRED, Philly Fed RTDSM), and international macro panels (WDI, OECD MEI, IMF IFS, Eurostat, ECB AWM). These provide the macro confounders and outcome series used in DML applications for monetary policy, business-cycle, and panel-macro inference.

---

## A1. 1-Year Expected Inflation (EXPINF1YR, Cleveland Fed model)

- **1-Year Expected Inflation (EXPINF1YR, Cleveland Fed model)** — Federal Reserve Bank of St. Louis (FRED) (1982).
  - **Source:** https://fred.stlouisfed.org/series/EXPINF1YR
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1982-01 onward; percent (modelled expectation); license: public domain (US gov) / FRED terms
  - **Tasks:** Federal Reserve Bank of Cleveland
  - **Status:** Verified.

## A2. 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (T10Y2Y)

- **10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (T10Y2Y)** — Federal Reserve Bank of St. Louis (FRED) (1976).
  - **Source:** https://fred.stlouisfed.org/series/T10Y2Y
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** daily, 1976-06 onward; percent (yield curve slope); license: public domain (US gov) / FRED terms
  - **Tasks:** Board of Governors of the Federal Reserve System, H.15
  - **Status:** Verified.

## A3. ALFRED: ArchivaL Federal Reserve Economic Data (Real-Time Vintages)

- **ALFRED: ArchivaL Federal Reserve Economic Data (Real-Time Vintages)** — Federal Reserve Bank of St. Louis (FRED) (2010).
  - **Source:** https://alfred.stlouisfed.org/
  - **Access:** API; auth_required: Y
  - **Schema:** task_family: time_series
  - **Size+License:** all FRED series with vintage history; thousands of series; license: public domain / FRED terms of use
  - **Tasks:** St. Louis Fed (ongoing)
  - **Status:** Unverified.

## A4. All Employees, Total Nonfarm (PAYEMS)

- **All Employees, Total Nonfarm (PAYEMS)** — Federal Reserve Bank of St. Louis (FRED) (1939).
  - **Source:** https://fred.stlouisfed.org/series/PAYEMS
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1939-01 onward; thousands of persons SA; license: public domain (US gov) / FRED terms
  - **Tasks:** U.S. Bureau of Labor Statistics, CES0000000001
  - **Status:** Verified.

## A5. Area-Wide Model (AWM) Database for the Euro Area

- **Area-Wide Model (AWM) Database for the Euro Area** — institutional / other (2001).
  - **Source:** https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp042.pdf
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** quarterly euro-area aggregates, 1970-Q1 onward; license: ECB working-paper release; redistribution unclear
  - **Tasks:** Fagan, Henry & Mestre (2001), ECB Working Paper 42
  - **Status:** Unverified.

## A6. BLS Public Data API (Bureau of Labor Statistics)

- **BLS Public Data API (Bureau of Labor Statistics)** — US Government open data (2024).
  - **Source:** https://www.bls.gov/developers/
  - **Access:** API; auth_required: Y
  - **Schema:** task_family: time_series
  - **Size+License:** thousands of CPI/CES/JOLTS/employment series; license: public domain (US gov) — BLS publishes everything in public domain except some copyrighted photographs
  - **Tasks:** U.S. Bureau of Labor Statistics
  - **Status:** Verified.

## A7. Bureau of Economic Analysis (BEA) Data Catalog (NIPA, ITAs, Regional)

- **Bureau of Economic Analysis (BEA) Data Catalog (NIPA, ITAs, Regional)** — US Government open data (2024).
  - **Source:** https://www.bea.gov/data
  - **Access:** API; auth_required: Y
  - **Schema:** task_family: time_series
  - **Size+License:** thousands of NIPA/GDP/regional/industry series; license: public domain (US gov)
  - **Tasks:** U.S. Bureau of Economic Analysis
  - **Status:** Verified.

## A8. CBOE Volatility Index: VIX (VIXCLS)

- **CBOE Volatility Index: VIX (VIXCLS)** — Federal Reserve Bank of St. Louis (FRED) (1990).
  - **Source:** https://fred.stlouisfed.org/series/VIXCLS
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** daily close, 1990-01 onward; license: CBOE redistribution restrictions + custom restrictions: cannot redistribute outside FRED viewer; check CBOE terms for derivative datasets
  - **Tasks:** Chicago Board Options Exchange via FRED
  - **Status:** Verified.

## A9. Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)

- **Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)** — Federal Reserve Bank of St. Louis (FRED) (1947).
  - **Source:** https://fred.stlouisfed.org/series/CPIAUCSL
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1947-01 onward; Index 1982-1984=100 SA; license: public domain (US gov) / FRED terms
  - **Tasks:** U.S. Bureau of Labor Statistics via FRED
  - **Status:** Verified.

## A10. Eurostat Database (EU statistical office)

- **Eurostat Database (EU statistical office)** — institutional / other (2024).
  - **Source:** https://ec.europa.eu/eurostat/web/main/data/database
  - **Access:** API; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** thousands of EU-harmonized series; SDMX API; license: Commission re-use policy (free re-use with attribution)
  - **Tasks:** European Commission - Eurostat
  - **Status:** Verified.

## A11. Federal Funds Effective Rate (DFF, daily)

- **Federal Funds Effective Rate (DFF, daily)** — Federal Reserve Bank of St. Louis (FRED) (1954).
  - **Source:** https://fred.stlouisfed.org/series/DFF
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** daily, 1954-07 onward; percent per annum; license: public domain (US gov) / FRED terms
  - **Tasks:** Board of Governors of the Federal Reserve System, H.15
  - **Status:** Verified.

## A12. Federal Funds Effective Rate (FEDFUNDS, monthly)

- **Federal Funds Effective Rate (FEDFUNDS, monthly)** — Federal Reserve Bank of St. Louis (FRED) (1954).
  - **Source:** https://fred.stlouisfed.org/series/FEDFUNDS
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1954-07 onward; percent per annum; license: public domain (US gov) / FRED terms
  - **Tasks:** Board of Governors of the Federal Reserve System, H.15
  - **Status:** Verified.

## A13. FRED API (Federal Reserve Economic Data Web Service)

- **FRED API (Federal Reserve Economic Data Web Service)** — Federal Reserve Bank of St. Louis (FRED) (2024).
  - **Source:** https://fred.stlouisfed.org/docs/api/fred/
  - **Access:** API; auth_required: Y
  - **Schema:** task_family: time_series
  - **Size+License:** 800,000+ series across 100+ sources; license: FRED Terms of Use (free; redistribution restrictions on some series)
  - **Tasks:** Federal Reserve Bank of St. Louis (2024)
  - **Status:** Unverified.

## A14. FRED-MD: A Monthly Database for Macroeconomic Research

- **FRED-MD: A Monthly Database for Macroeconomic Research** — Federal Reserve Bank of St. Louis (FRED) (2015).
  - **Source:** https://research.stlouisfed.org/econ/mccracken/fred-databases/
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** ~128 monthly series, 1959-01 onward, ~1 MB CSV per vintage; license: FRED terms of use (free with proprietary-notice preservation); R-package mirrors (BVAR, fbi) re-license as modified ODC-BY 1.0
  - **Tasks:** McCracken & Ng (2016), Journal of Business & Economic Statistics 34(4)
  - **Status:** Unverified.

## A15. FRED-QD: A Quarterly Database for Macroeconomic Research

- **FRED-QD: A Quarterly Database for Macroeconomic Research** — Federal Reserve Bank of St. Louis (FRED) (2021).
  - **Source:** https://research.stlouisfed.org/econ/mccracken/fred-databases/
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** ~248 quarterly series, 1959-Q1 onward, vintages from 2018-05; license: FRED terms of use (free with proprietary-notice preservation); R-package mirrors re-license as modified ODC-BY 1.0
  - **Tasks:** McCracken & Ng (2020), Federal Reserve Bank of St. Louis Review
  - **Status:** Unverified.

## A16. Gross Domestic Product: Implicit Price Deflator (GDPDEF)

- **Gross Domestic Product: Implicit Price Deflator (GDPDEF)** — Federal Reserve Bank of St. Louis (FRED) (1947).
  - **Source:** https://fred.stlouisfed.org/series/GDPDEF
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** quarterly, 1947-Q1 onward; Index 2017=100 SA; license: public domain (US gov) / FRED terms
  - **Tasks:** U.S. Bureau of Economic Analysis, NIPA
  - **Status:** Verified.

## A17. IMF International Financial Statistics (IFS)

- **IMF International Financial Statistics (IFS)** — institutional / other (2024).
  - **Source:** https://data.imf.org/ifs
  - **Access:** API; auth_required: Y
  - **Schema:** task_family: time_series
  - **Size+License:** thousands of monetary/balance-of-payments/exchange series for ~200 countries; license: IMF Terms of Use (free download; registration to save/download)
  - **Tasks:** International Monetary Fund
  - **Status:** Unverified.

## A18. Industrial Production: Total Index (INDPRO)

- **Industrial Production: Total Index (INDPRO)** — Federal Reserve Bank of St. Louis (FRED) (1919).
  - **Source:** https://fred.stlouisfed.org/series/INDPRO
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1919-01 onward; Index 2017=100 SA; license: public domain (US gov) / FRED terms
  - **Tasks:** Board of Governors of the Federal Reserve System, G.17
  - **Status:** Verified.

## A19. M2 (M2SL, seasonally adjusted)

- **M2 (M2SL, seasonally adjusted)** — Federal Reserve Bank of St. Louis (FRED) (1959).
  - **Source:** https://fred.stlouisfed.org/series/M2SL
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1959-01 onward; billions of USD SA; license: public domain (US gov) / FRED terms
  - **Tasks:** Board of Governors of the Federal Reserve System, H.6
  - **Status:** Verified.

## A20. Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity (DGS10)

- **Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity (DGS10)** — Federal Reserve Bank of St. Louis (FRED) (1962).
  - **Source:** https://fred.stlouisfed.org/series/DGS10
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** daily, 1962-01 onward; percent per annum; license: public domain (US gov) / FRED terms
  - **Tasks:** Board of Governors of the Federal Reserve System, H.15
  - **Status:** Verified.

## A21. Nasdaq Data Link (formerly Quandl) — Economic Datasets

- **Nasdaq Data Link (formerly Quandl) — Economic Datasets** — institutional / other (2024).
  - **Source:** https://data.nasdaq.com/
  - **Access:** API; auth_required: Y
  - **Schema:** task_family: time_series
  - **Size+License:** license: mixed: many free (World Bank/UN/Eurostat/BEA/BLS/FRED mirrors) under publisher terms + paid premium datasets
  - **Tasks:** Nasdaq, Inc. (Quandl rebrand 2021)
  - **Status:** Unverified.

## A22. NBER Macrohistory Database

- **NBER Macrohistory Database** — institutional / other (2024).
  - **Source:** https://www.nber.org/research/data/nber-macrohistory-database
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** ~3,036 historical series (pre-WWI + interwar US, partial UK/FR/DE); license: public (research use)
  - **Tasks:** NBER (curated; original assembly by Wesley Mitchell et al.)
  - **Status:** Verified.

## A23. NBER-based Recession Indicators for the United States (USREC)

- **NBER-based Recession Indicators for the United States (USREC)** — Federal Reserve Bank of St. Louis (FRED) (n.d.).
  - **Source:** https://fred.stlouisfed.org/series/USREC
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly binary {0,1}, 1854-12 onward; license: public (NBER) / FRED terms
  - **Tasks:** NBER Business Cycle Dating Committee via FRED
  - **Status:** Verified.

## A24. New Privately-Owned Housing Units Started: Total Units (HOUST)

- **New Privately-Owned Housing Units Started: Total Units (HOUST)** — Federal Reserve Bank of St. Louis (FRED) (1959).
  - **Source:** https://fred.stlouisfed.org/series/HOUST
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1959-01 onward; thousands of units SAAR; license: public domain (US gov) / FRED terms
  - **Tasks:** U.S. Census Bureau & HUD
  - **Status:** Verified.

## A25. OECD Main Economic Indicators (MEI) — Data Explorer / SDMX

- **OECD Main Economic Indicators (MEI) — Data Explorer / SDMX** — institutional / other (2024).
  - **Source:** https://data-explorer.oecd.org/
  - **Access:** API; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** short-term indicators across OECD + selected non-members; license: OECD Terms and Conditions (free, attribution required)
  - **Tasks:** Organisation for Economic Co-operation and Development
  - **Status:** Verified.

## A26. Penn World Table (PWT)

- **Penn World Table (PWT)** — institutional / other (n.d.).
  - **Source:** https://www.rug.nl/ggdc/productivity/pwt/
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** real GDP/TFP/capital across 183 countries, 1950 onward; license: CC-BY-4.0
  - **Tasks:** Feenstra, Inklaar & Timmer (2015), American Economic Review 105(10)
  - **Status:** Verified.

## A27. Personal Consumption Expenditures: Chain-type Price Index (PCEPI)

- **Personal Consumption Expenditures: Chain-type Price Index (PCEPI)** — Federal Reserve Bank of St. Louis (FRED) (1959).
  - **Source:** https://fred.stlouisfed.org/series/PCEPI
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1959-01 onward; Index 2017=100 SA; license: public domain (US gov) / FRED terms
  - **Tasks:** U.S. Bureau of Economic Analysis, Personal Income & Outlays
  - **Status:** Verified.

## A28. Real Gross Domestic Product (GDPC1)

- **Real Gross Domestic Product (GDPC1)** — Federal Reserve Bank of St. Louis (FRED) (1947).
  - **Source:** https://fred.stlouisfed.org/series/GDPC1
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** quarterly, 1947-Q1 onward; billions of chained 2017 USD SAAR; license: public domain (US gov) / FRED terms
  - **Tasks:** U.S. Bureau of Economic Analysis, NIPA
  - **Status:** Verified.

## A29. Real-Time Data Set for Macroeconomists (RTDSM)

- **Real-Time Data Set for Macroeconomists (RTDSM)** — institutional / other (2001).
  - **Source:** https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** vintages updated monthly; major NIPA aggregates from 1965 onward; license: public (Philadelphia Fed; research use)
  - **Tasks:** Croushore & Stark (2001), Journal of Econometrics 105
  - **Status:** Verified.

## A30. Spot Crude Oil Price: West Texas Intermediate (WTISPLC)

- **Spot Crude Oil Price: West Texas Intermediate (WTISPLC)** — Federal Reserve Bank of St. Louis (FRED) (1946).
  - **Source:** https://fred.stlouisfed.org/series/WTISPLC
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1946-01 onward; USD per barrel; license: public domain (US gov composite) / FRED terms
  - **Tasks:** St. Louis Fed (combines OILPRICE + MCOILWTICO)
  - **Status:** Verified.

## A31. U.S. Census Bureau Data API (data.census.gov)

- **U.S. Census Bureau Data API (data.census.gov)** — US Government open data (2024).
  - **Source:** https://www.census.gov/data/developers.html
  - **Access:** API; auth_required: Y
  - **Schema:** task_family: tabular
  - **Size+License:** ACS, decennial census, economic census, building permits, etc.; license: public domain (US gov)
  - **Tasks:** U.S. Census Bureau
  - **Status:** Unverified.

## A32. U.S. Dollars to Euro Spot Exchange Rate (DEXUSEU)

- **U.S. Dollars to Euro Spot Exchange Rate (DEXUSEU)** — Federal Reserve Bank of St. Louis (FRED) (1999).
  - **Source:** https://fred.stlouisfed.org/series/DEXUSEU
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** daily, 1999-01-04 onward; USD per 1 EUR (noon NYC); license: public domain (US gov) / FRED terms
  - **Tasks:** Board of Governors of the Federal Reserve System, H.10
  - **Status:** Verified.

## A33. Unemployment Rate (UNRATE)

- **Unemployment Rate (UNRATE)** — Federal Reserve Bank of St. Louis (FRED) (1948).
  - **Source:** https://fred.stlouisfed.org/series/UNRATE
  - **Access:** direct; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** monthly, 1948-01 onward; percent SA; license: public domain (US gov) / FRED terms
  - **Tasks:** U.S. Bureau of Labor Statistics, CPS LNS14000000
  - **Status:** Verified.

## A34. World Development Indicators (WDI)

- **World Development Indicators (WDI)** — institutional / other (2024).
  - **Source:** https://datatopics.worldbank.org/world-development-indicators/
  - **Access:** API; auth_required: N
  - **Schema:** task_family: time_series
  - **Size+License:** ~1,486 indicators x 266 countries/entities; annual 1960 onward; license: CC-BY-4.0
  - **Tasks:** World Bank Group
  - **Status:** Verified.

