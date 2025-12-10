# Predicting Air Temperature Using Beach Sensor Data

## Executive Summary

This report analyzes weather data using beach sensor data, which covers 196,516 hourly measurements between January 2016 to December 2024 from Lake Micigan beaches in Chicago. The project consists of 9-phases workflow to understand patterns in beach weather conditions and build predictive models for air temperature with the XGBoost model and linear regression model. Some of the key findings include strong seasonal temperature patterns, meaning summer months are warmer and winter months are colder, where there are also significant daily cycles. The XGBoost model emerged as the best performer, with a test R² of 0.9998 and RMSE of 0.146, which show that air temperature can be predicted with good accuracy from features such as temperature to humidity ratio as well as temperature with humidity interaction.

## Phase-by-Phase Findings

### Phase 1-2: Exploration

From the initial exploration we know the dataset contains 196,516 records with 18 columns including temperature measurements (air and wet bulb), wind speed and direction, humidity, precipitation type, barometric pressure, solar radiation, heading, battery life, and rain information data. The data ranges from January 2016 to December 2024, with measurements from three different weather stations; 63rd Street Weather Station, Foster Weather Station, and Oak Street Weather Station. 

**Key Data Quality Issues Identified:**
- Around 75 missing values for Air Temperature (0.0%), which is minimal and less than 5.0% of the total data.
- Approximately 76,049 missing values in Wet Bulb Temperature (38.7%), Rain Intensity (38.7%), Precipitation Type (38.7%), and Heading (38.7%).
- Approximately 146 missing values in Barometric Pressure (0.1%).
- 146 (0.1%) missing values in Barometric Pressure
- Some outliers in olar Radiation and Heading
- Data collected at hourly intervals with some gaps
  
Initial visualizations showed:
- Air temperature ranging from -29.78°C to 37.6°C
- Air temperature has been identical over the years with small amount of variability
- Clear seasonal patterns visible in temperature data
- No noticeable long term warming or cooling trend is visible across the 9-year period.
- The distribution has two peaks with a peak around late fall, and another peak around summer months.
  
![Figure 1: Initial Data Exploration](output/q1_visualizations.png)
*Figure 1: Initial exploration visualizations showing distributions of air temperature frequencies and air temperature over the years.*

### Phase 3: Data Cleaning

Data cleaning focused on resolving missing values, correcting data types, handling outliers, and ensuring dataset consistency without removing rows. Missing values were handled using methods appropriate to each variable type. Air Temperature and Barometric Pressure—both smooth, continuous time-series variables—were imputed using interpolation to preserve temporal continuity. Variables with extensive missingness such as Wet Bulb Temperature and Heading were filled using median imputation, which avoids introducing patterns in cases where large segments of data are absent. Rain-related fields (Rain Intensity, Total Rain) were filled with 0, and the categorical variable Precipitation Type was filled with "None", reflecting reasonable domain assumptions for missing precipitation values.

**Cleaning Results:**
- Rows before cleaning: **196,516**
- Missing values: Interpolated and median-imputed
  - `Air Temperature`: 75 missing → 0 missing (interpolated)
  - `Wet Bulb Temperature`: 76,049 missing → 0 missing (median imputed)
  - `Barometric Pressure`: 146 missing → 0 missing (interpolated)
  - `Total Rain` — 76,049 missing → 0 (filled with 0)
  - `Precipitation Type` — 76,049 missing → 0 (filled with "None")
- Outliers: Capped using IQR method (1.5×IQR bounds)
  - Wind Speed: 12,225 outliers capped (bounds: [-3.50, 8.40])
- Duplicates: Removed (0 duplicates found)
- Rows after cleaning: **196,516** (no rows removed, only values cleaned)

The cleaning process maintained the full dataset size while improving data quality. The large number of missing values in Wet Bulb Temperature (38.7%) likely reflects sensor downtime or unavailable measurements at certain stations, but imputation strategies ensured these features remained usable for later analysis and modeling.

### Phase 4: Data Wrangling

Datetime parsing and temporal feature extraction were critical for time series analysis. The `Measurement Timestamp` column was parsed from the format "MM/DD/YYYY HH:MM:SS AM/PM" and set as the DataFrame index, enabling time-based operations.

**Temporal Features Extracted:**
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 5=Saturday)
- `month`: Month of year (1-12)
- `year`: Year
- `day_name`: Day name (Monday-Sunday)
- `is_weekend`: Binary indicator (1 if Saturday/Sunday)

The dataset covers roughly 9 years of continuous hourly measurements (January 2016 to December 2024), which provides sufficient granularity and duration to support seasonal trend analysis, feature engineering, and robust forecasting.

### Phase 5: Feature Engineering

Feature engineering created derived variables and rolling window statistics to capture relationships and temporal dependencies.

**Derived Change Features**:
- `AirTemp_Change`: Hour-to-hour change in air temperature
- `Humidity_Change`: Hour-to-hour change in humidity
- `Pressure_Change`: Hour-to-hour change in barometric pressure
- `WindSpeed_Change`: Hour-to-hour change in wind speed

**Interaction and Ratio Features**:
- `Wind_Gust_Ratio`: Ratio of maximum wind speed to sustained wind speed
- `Rain_Per_Interval`: Rainfall normalized by interval length
- `Temp_to_Humidity_Ratio`: “Humidity-normalized” temperature indicator
- `Temp_Humidity_Interaction`: Temperature × humidity interaction term
- `Wind_Solar_Interaction`: Wind speed × solar radiation interaction

**Rolling Window Features**:
Time-based rolling windows were added to capture short-term temporal smoothing:
- `wind_speed_rolling_7h`: 7-hour rolling mean of wind speed
- `humidity_rolling_24h`: 24-hour rolling mean of humidity
- `pressure_rolling_7h`: 7-hour rolling mean of barometric pressure
