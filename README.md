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
- Python 3.12.3
- Main libraries: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `contextily`, `tensorflow`, `keras`, `joblib`

### ⚙️ Configuration

⚠️ **Important:** Before running the model, you must edit the configuration section in  
`superficie_modelo_5_1.py` and verify that all file paths are correct. The following inputs must be properly linked:

- FNN models for wind field forecasting  
- FNN model for mobile source emissions  
- ProAire and DENUE emission inventories  
- Meteorological data (INIFAP/RUOA)  
- SINAICA air quality data (for validation)  

Paths can be modified directly in the script `superficie_modelo_5_1.py`.
