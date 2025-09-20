# Hybrid Modular PM<sub>2.5</sub> Forecasting Model

This repository contains the implementation of a **hybrid modular model** for forecasting PM<sub>2.5</sub> concentrations in Aguascalientes, Mexico. The model integrates **mass transport equations** with **feedforward neural networks (FNNs)** to simulate air pollutant dynamics in cities with sparse monitoring networks.

## Features
- Hybrid approach: phenomenological transport + machine learning
- Three main modules:
  1. **Wind fields:** FNN trained with INIFAP and RUOA meteorological data  
  2. **Mixing layer height:** estimated from solar radiation, temperature, and wind speed  
  3. **Emissions:**  
     - Mobile sources: FNN trained with vehicular traffic data  
     - Fixed/area sources: ProAire inventory + INEGI DENUE database  
- Outputs integrated into a discretized transport solver for 24h forecasts  

## Requirements
- Python 3.11+
- Main libraries: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `geopandas`

Install dependencies:
```bash
pip install -r requirements.txt
