import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Devyansh verma\OneDrive\Desktop\=excel\hjkl\devya\COVID_ANALYSIS new version\COVID_ANALYSIS\data\Provisional_COVID-19_Deaths_by_Sex_and_Age.csv")
print(df.head())
print("Shape:", df.shape)
print("\nInfo:\n", df.info())
print("\nMissing values:\n", df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns

rows_with_note = df[~df['Footnote'].isnull()].index
df.loc[rows_with_note, numeric_cols] = df.loc[rows_with_note, numeric_cols].fillna(5)

df.isnull().sum()

df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Month'] = df['Start Date'].dt.month_name()
df['Year'] = df['Start Date'].dt.year

category_cols = ['Sex', 'Age Group', 'State', 'Month', 'Year']
df[category_cols] = df[category_cols].astype('category')

df.info()

#EDA

print("Summary stats for COVID-19 Deaths: \n",df['COVID-19 Deaths'].describe())
print("\nPneumonia Deaths: \n",df['Pneumonia Deaths'].describe())
print("\nInfluenza Deaths: \n",df['Influenza Deaths'].describe())
print("\nTotal Deaths: \n",df['Total Deaths'].describe())

#Plots

sns.kdeplot(np.log1p(df['COVID-19 Deaths']))
plt.title("Distribution of COVID-19 Deaths (Log Scale)")
plt.xlabel("log(COVID-19 Deaths + 1)")
plt.show()

sns.kdeplot(np.log1p(df['Pneumonia Deaths']))
plt.title("Distribution of Pneumonia Deaths (Log Scale)")
plt.xlabel("log(Pneumonia Deaths + 1)")
plt.show()

sns.kdeplot(np.log1p(df['Influenza Deaths']))
plt.title("Distribution of Influenza Deaths (Log Scale)")
plt.xlabel("log(Influenza Deaths + 1)")
plt.show()


plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Month', palette="viridis", edgecolor="black")
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.title("Count of Records by Month", fontsize=16, fontweight="bold")
plt.xlabel("Month", fontsize=12)
plt.ylabel("Record Count", fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.4)  
plt.tight_layout()
plt.show()


#Multivariate analysis
plt.figure(figsize=(10, 7), dpi=120)
corr = df[['COVID-19 Deaths','Pneumonia Deaths','Influenza Deaths']].corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    linewidths=0.7,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Correlation Strength"}
)

plt.title("Correlation Matrix of Death Causes", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

plt.style.use("dark_background")

#Obj -1: COVID Deaths by Age Group & Sex
plt.style.use("dark_background")
df_age_sex = df.groupby(['Age Group','Sex'])['COVID-19 Deaths'].sum().reset_index()
g = sns.catplot(
    data=df_age_sex,
    x='Age Group',
    y='COVID-19 Deaths',
    col='Sex',
    kind='bar',
    height=5,
    col_wrap=2,
    palette="viridis"
)

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', color="white")
    ax.set_ylabel("COVID-19 Deaths", color="white")
    ax.set_xlabel("Age Group", color="white")
    ax.tick_params(colors="white")
    ax.grid(axis='y', linestyle='--', alpha=0.2)

g.fig.suptitle("COVID-19 Deaths by Age Group & Sex", fontsize=16, color="white")
g.fig.tight_layout()
plt.show()


#Obj -2: Sex-Based Mortality
plt.style.use("seaborn-v0_8")

df_sex = df.groupby('Sex')['Total Deaths'].sum().reset_index()

plt.figure(figsize=(8,5))
ax = sns.barplot(
    data=df_sex,
    x='Sex',
    y='Total Deaths',
    palette='viridis',
    edgecolor='black'
)
for i, row in df_sex.iterrows():
    ax.text(
        i,
        row['Total Deaths'] + (row['Total Deaths'] * 0.02),
        f"{int(row['Total Deaths']):,}",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

plt.title("Total Deaths by Sex", fontsize=16, weight='bold')
plt.xlabel("Sex", fontsize=12)
plt.ylabel("Total Deaths", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

#Obj -3: Compare COVID, Pneumonia & Influenza
df_total = df[['COVID-19 Deaths','Pneumonia Deaths','Influenza Deaths']].sum()

plt.figure(figsize=(7,7))

colors = sns.color_palette("viridis", 3)
explode = [0.05, 0.05, 0.05]

plt.pie(
    df_total.values,
    labels=df_total.index,
    autopct='%1.1f%%',
    explode=explode,
    colors=colors,
    shadow=True,
    startangle=140,
    textprops={'fontsize': 12}
)

centre_circle = plt.Circle((0,0), 0.60, fc='white')
plt.gca().add_artist(centre_circle)

plt.title("Overall Cause-wise Death Distribution", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


#Obj -4: Peak Month for Each State
df['State'] = df['State'].str.strip().str.title()

exclude_list = [
    "United States",
    "District Of Columbia",
    "Puerto Rico",
    "Guam",
    "American Samoa",
    "Northern Mariana Islands",
    "Virgin Islands"
]
df = df[~df['State'].isin(exclude_list)]

# Compute peak month for each state
df_state_peak = (
    df.groupby(['State','Year','Month'])['COVID-19 Deaths']
      .sum()
      .reset_index()
      .sort_values('COVID-19 Deaths', ascending=False)
)

df_state_peak_top = df_state_peak.drop_duplicates('State')
df_top10 = df_state_peak_top.head(10).sort_values('COVID-19 Deaths')

# --- BEAUTIFUL MULTICOLOR CHART ---
plt.figure(figsize=(14, 10))

sns.set_theme(style="white", rc={
    "axes.facecolor": "#f7f7f7",
    "grid.color": "#dcdcdc",
    "axes.edgecolor": "#333",
    "font.size": 12
})

# Unique fancy colors for each bar
unique_colors = sns.color_palette("Spectral", len(df_top10))

ax = sns.barplot(
    data=df_top10,
    y="State",
    x="COVID-19 Deaths",
    palette=unique_colors,
    edgecolor="black",
    linewidth=1.2
)

# Title
plt.title(
    "Top 10 U.S. States With Highest Peak COVID-19 Deaths\n"
    "(Each Bar = Unique Color, Peak Month Labeled)",
    fontsize=22,
    weight='bold',
    pad=20
)

plt.xlabel("Peak COVID-19 Deaths", fontsize=15)
plt.ylabel("State", fontsize=15)

# Value labels + Month labels on bars
for index, p in enumerate(ax.patches):
    width = p.get_width()
    state = df_top10.iloc[index]
    month = state["Month"]

    ax.text(
        width + (max(df_top10["COVID-19 Deaths"]) * 0.015),
        p.get_y() + p.get_height()/2,
        f"{int(width):,}  ({month})",
        va='center',
        fontsize=12,
        color="#333"
    )

# Remove legend entirely (month already displayed beside bars)
plt.legend([], [], frameon=False)

plt.tight_layout()
plt.show()


#Obj -5: COVID Deaths as % of Total Deaths

covid_percent = (df['COVID-19 Deaths'].sum() / df['Total Deaths'].sum()) * 100
print("COVID Deaths as % of Total Deaths: ",covid_percent)

#Obj -6: Monthly Trend Analysis
df_yearly = df.groupby(['Year','Month'])['COVID-19 Deaths'].sum().reset_index()

plt.figure(figsize=(14,7))
sns.set_theme(style="whitegrid")

for year in df_yearly['Year'].unique():
    data = df_yearly[df_yearly['Year'] == year]
    plt.plot(data['Month'], data['COVID-19 Deaths'], marker='o', linewidth=2, label=str(year))

plt.xticks(rotation=45)
plt.title("Month-wise COVID-19 Death Trend (Separated by Year)", fontsize=18, weight="bold")
plt.xlabel("Month")
plt.ylabel("COVID-19 Deaths")
plt.legend(title="Year")

plt.tight_layout()
plt.show()