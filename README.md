# Covid-19-Deaths-Analysis
📊 COVID-19 Mortality Data Analysis (Python)

This project performs an end-to-end exploratory data analysis (EDA) on the Provisional COVID-19 Deaths by Sex and Age dataset.
It focuses on cleaning, transforming, visualizing, and deriving insights from U.S. mortality data related to COVID-19, Pneumonia, and Influenza, including demographic and state-wise patterns.

🛠️ Tools & Libraries Used
Pandas – data cleaning, manipulation, grouping
NumPy – numerical operations
Matplotlib – basic visualizations
Seaborn – advanced and styled plots
Datetime (built-in) – date parsing & time-based transformations

📁 Key Project Steps (Concise Version)

✔ 1. Data Cleaning

- Handled missing values (special rule for rows with footnotes)
- Parsed date columns and standardized category fields
- Removed non-state territories for accurate state-level analysis

✔ 2. Feature Engineering

- Extracted Month and Year from Start Date
- Converted key columns (Sex, Age Group, State) to categorical types
- Created grouped datasets for demographic and state-wise analysis

✔ 3. Exploratory Data Analysis

- Distribution plots for COVID/Pneumonia/Influenza deaths (log-scaled)
- Monthly record counts and yearly COVID trends
- Correlation heatmap for major death causes

✔ 4. Visualization Insights

- Age group × Sex COVID mortality comparison
- Total deaths by sex
- Cause-wise comparison using a donut chart
- Top 10 states with highest peak COVID deaths (labeled with month)

📈 Key Insights

- Identified the most affected age groups and sex categories
- Found peak COVID-19 months for each U.S. state
- Observed strong correlation between COVID-19 and Pneumonia deaths
- Highlighted month-wise trends across multiple years
- Calculated COVID-19 deaths as a percentage of total deaths
