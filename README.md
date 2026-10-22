# 🌍 Global Development Clusters

Hierarchical clustering of **167 countries** on 9 socio-economic indicators, with an interactive **Streamlit dashboard** that shows which countries need aid the most.

> 🔗 **Live demo:** _add your Streamlit link here_

## Screenshots

_Add 3 to 4 screenshots here (world map, dendrogram, cluster profiles, aid priority)._

## What this project does

1. **Explores** the data: distributions, skewness and correlations
2. **Preprocesses** it: log/sqrt transforms for skewed features, then standardization
3. **Compares 4 linkage methods** (Ward, complete, average, single) using cophenetic correlation and silhouette score
4. **Clusters** the countries with Ward linkage (k = 4)
5. **Ranks** groups and countries with a simple *need score*
6. **Visualizes** everything in an interactive dashboard

## Key findings

- **Ward linkage** was chosen even though average linkage had a higher silhouette score. Average linkage produced clusters of 161, 5 and 1 countries, which is not useful for decisions. Ward gave balanced, interpretable groups (56, 61, 45 and 5 countries).
- **The 5 smallest-cluster countries are trade hubs** (Singapore, Luxembourg, Malta, Ireland, Seychelles) whose exports and imports exceed 100% of GDP.
- **Highest-need countries** in the data include Congo (Dem. Rep.), Burundi and Sierra Leone.

| Group | Countries | Description |
|---|---|---|
| Group 1 | 56 | Highest need: high child mortality, low income and life expectancy |
| Group 2 | 61 | Developing countries, moderate need |
| Group 3 | 45 | Developed countries, low need |
| Group 4 | 5 | Wealthy trade hubs, lowest need |

## Dashboard features

- 🗺️ World map coloured by cluster
- 🌳 Dendrogram with a live **k slider**
- 📊 Group profiles (heatmap and radar chart)
- 🔎 Country explorer with similar-country search
- 🎯 Ranked aid-priority list with CSV download

## Tech stack

Python · pandas · NumPy · SciPy · scikit-learn · Plotly · Matplotlib · Streamlit

## Project structure

```
├── training.ipynb            # full analysis and model building
├── app.py                    # Streamlit dashboard
├── countries_clustered.csv   # clustered output used by the dashboard
├── models/                   # saved linkage matrix, scaler and metrics
├── .streamlit/config.toml    # dark theme
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/<your-username>/country-clustering.git
cd country-clustering
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

To retrain, download **Country-data.csv** from Kaggle ("Unsupervised Learning on Country Data"), place it in a `data/` folder and run `training.ipynb`.

## Dataset

Kaggle: *Unsupervised Learning on Country Data*. Features: child mortality, exports, health spending, imports, income, inflation, life expectancy, total fertility and GDP per capita.
