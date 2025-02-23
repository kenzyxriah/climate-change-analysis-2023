import streamlit as st
import pandas as pd
import math
from pathlib import Path
import pandas as pd
import streamlit as st
import math
from pathlib import Path
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
import plotly.graph_objects as go

df = pd.read_csv(r'C:\Users\Admin\Documents\Data Science\Python\climate_change_p\climate_change_mod.csv')

# Streamlit App Title
st.title("🌍 Climate Change Impact and Risk Assessment")

# Introduction Section
st.markdown("""
## 📌 Project Overview  
Climate change is one of the most pressing global challenges, affecting ecosystems, economies, and human livelihoods. Rising temperatures, increasing CO₂ emissions, deforestation, and extreme weather events are contributing to a complex web of environmental issues.  
This project provides a **data-driven approach** to understanding climate change’s impact across different continents and countries using **interactive visualizations and statistical analysis**.  

## 🔍 Why This Analysis?  
The purpose of this analysis is to:  

1️⃣ **Track Global Climate Trends** 📊  
   - Analyze how key climate variables (e.g., temperature, CO₂ emissions, rainfall, sea level rise) have changed over time.  
   - Provide a **visual understanding** of climate patterns and their **regional variations**.  

2️⃣ **Assess Climate-Related Risks** ⚠️  
   - Rank countries based on climate impact indicators using a **composite risk score**.  
   - Identify regions that are most vulnerable to climate change effects.  

3️⃣ **Understand Relationships Between Climate Variables** 🔬  
   - Explore correlations between **CO₂ emissions, temperature, deforestation, and extreme weather events**.  
   - Perform statistical tests (ANOVA) to examine differences across continents.  

4️⃣ **Enable Interactive Data Exploration** 🎛️  
   - Select and filter data dynamically using Streamlit’s interactive widgets.  
   - Compare climate trends across different continents and timeframes.  

By providing these insights, this project aims to **enhance awareness** and encourage **data-driven climate action**. 🌱🌎  
""")

# -----------------------------------------------------------------------------
# Declare some useful functions.
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\Admin\Documents\Data Science\Python\climate_change_p\climate_change_mod.csv')  

df = load_data()

# Define columns
cat_cols = ['Year', 'Country', 'ISO Country Code', 'Continent']
numeric_cols = ['Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 
                'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 'Extreme Weather Events',
                'Forest Area (%)', 'Rainfall (mm)_per_100', 'Pop_Density(per_100m)']


st.write("First two rows of the dataset:")
st.dataframe(df[cat_cols + numeric_cols].head(5))

# Density Plot Section
st.subheader("KDE Density Plot")

# User selection
x = st.selectbox("Select the numerical column:", numeric_cols)
y = st.selectbox("Select the categorical column for grouping:", cat_cols)
include_hist = st.checkbox("Include Histogram", value=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=df, x=x, hue=y, fill=False, common_norm=False, ax=ax)

if include_hist:
    ax1 = ax.twinx()
    sns.histplot(data=df, x=x, hue=y, element="step", common_norm=False, alpha=0.3, ax=ax1)

ax.set_title("KDE Density Plot with Optional Histogram")
ax.set_xlabel(x)
ax.set_ylabel("Density")
ax.legend(title=y)

st.pyplot(fig)

# Line Trend Plot Section
st.subheader("Line Trend Analysis")
trend_x = st.selectbox("Select the numerical column for trend analysis:", numeric_cols, key='trend_x')
year = df.groupby('Year', as_index=False)[trend_x].mean()
year['pct_change'] = year[trend_x].pct_change() * 100

# Create the plot
fig2, ax2 = plt.subplots(figsize=(12, 5))
sns.lineplot(x='Year', y=trend_x, data=year, ax=ax2, color='k', marker='o', linestyle='--', markerfacecolor='red', errorbar=None)

# Annotate percentage change
for i, value in enumerate(year['pct_change']):
    if pd.notna(value):  
        ax2.text(year['Year'].iloc[i], year[trend_x].iloc[i], f'{value:.2f}%', 
                 ha='center', color='black', fontsize=10)

plt.axvspan(2013, 2023, color="orange", alpha=0.3, label="Last Decade")
ax2.set_xlabel('Year')
ax2.set_ylabel(trend_x)
plt.title(f'Trend and Percentage Change of {trend_x} Over Time')

include_projection = st.checkbox("Include Projection Trend", value=False, key='projection')

if include_projection:
    X = year['Year'].values.reshape(-1, 1)
    y = year[trend_x].values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    future_years = np.array([year['Year'].max() + i for i in range(1, 11)]).reshape(-1, 1)
    future_years_with_const = sm.add_constant(future_years)
    future_temp_predictions = model.predict(future_years_with_const)
    predictions_with_ci = model.get_prediction(future_years_with_const)
    ci_lower, ci_upper = predictions_with_ci.conf_int(alpha=0.05).T
    
    plt.fill_between(future_years.flatten(), ci_lower, ci_upper, color='gray', alpha=0.3, label='Confidence Interval (95%)')
    plt.plot(future_years, future_temp_predictions, label='Projections', color='red', linestyle='--')
    
    # Model evaluation using MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y - model.predict(X)))
    accuracy = 100 - (mae / np.mean(y) * 100)

    st.write(f"Mean Absolute Error (MAE) on training data: {mae:.2f}")
    st.write(f"Model Accuracy: {accuracy:.2f}%")

    if accuracy >= 85:
        st.write("The model has reached a minimum accuracy of 85%")
    else:
        st.write("The model has not reached a minimum accuracy of 85%")

st.pyplot(fig2)


st.subheader("Line Plot: Trend Over Time")

# Select numerical column for y-axis
x = st.selectbox("Select numerical column:", numeric_cols, key='lineplot_x')

# Select categorical column for filtering
y = st.selectbox("Filter by category:", cat_cols, key='lineplot_y')

# Get top 5 categories based on mean value of the selected numerical column
top_5_categories = df.groupby(y)[x].mean().nlargest(5).index.tolist()

# Allow user to select specific categories
selected_categories = st.multiselect(f"Select specific {y} values (default: top 5)", 
                                     df[y].unique(), default=top_5_categories)

# Year range slider
min_year, max_year = df["Year"].min(), df["Year"].max()
year_range = st.slider("Select year range:", min_year, max_year, (min_year, max_year))

# Filter data based on selections
filtered_df = df[(df[y].isin(selected_categories)) & (df["Year"].between(*year_range))]

# Plot the line plot
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=filtered_df, x="Year", y=x, hue=y, marker="o", ax=ax3, errorbar = None)

ax3.set_title(f"Trend of {x} Over Time (Filtered by {y})")
ax3.set_xlabel("Year")
ax3.set_ylabel(x)
ax3.legend(title=y)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), ncol=6)

st.pyplot(fig3)

# Bar Plot Section
st.subheader("Bar Plot: Numerical vs Year")
bar_x = st.selectbox("Select the numerical column for bar plot:", numeric_cols, key='bar_x')
stack_by_continent = st.checkbox("Stack by Continent", value=False, key='stack_continent')

colors = ['#a85c32', '#0c1a42', '#3b3430', '#420c24', '#0c3c42', 'brown']

# Year Range Slider
year_range = st.slider("Select Year Range:", int(df['Year'].min()), int(df['Year'].max()), (int(df['Year'].min()), int(df['Year'].max())))
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

fig4, ax4 = plt.subplots(figsize=(12, 6))
if stack_by_continent:
    sns.barplot(data=df_filtered, x='Year', y=bar_x, hue='Continent', ax=ax4, dodge=False, palette = colors, errorbar = None)
    plt.legend(title='Category')
    # Display legend at the bottom
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=6)
    
else:
    sns.barplot(data=df_filtered, x='Year', y=bar_x, ax=ax4, color='blue', errorbar = None)

ax4.set_title(f"Bar Plot of {bar_x} Over Time")
ax4.set_xlabel("Year")
ax4.set_ylabel(bar_x)

# Rotate x-axis labels for readability
plt.xticks(rotation=90)

st.pyplot(fig4)


st.subheader("Scatter Plot (with Regression)")
x_col = st.selectbox("Select X-axis (Numerical):", numeric_cols, key="X")
y_col = st.selectbox("Select Y-axis (Numerical):", numeric_cols, key="scatter_y")
plot_by = st.radio("Plot By:", ["Continent", "All Data"], key="scatter_plot_by")

# Sort data by Year
df_plot = df.sort_values(by='Year', ascending=True)

# Compute correlation per continent per year
if plot_by == "Continent":
    correlation_per_group = df_plot.groupby(['Year', 'Continent']).apply(
        lambda x: x[[x_col, y_col]].corr().iloc[0, 1] if len(x) > 1 else None
    ).reset_index(name='Correlation')
else:
    correlation_per_group = df_plot.groupby(['Year']).apply(
        lambda x: x[[x_col, y_col]].corr().iloc[0, 1] if len(x) > 1 else None
    ).reset_index(name='Correlation')

# Merge correlation data
df_plot = df_plot.merge(correlation_per_group, on=['Year'] + (['Continent'] if plot_by == "Continent" else []), how='left')

# Assign correlation text only to the first occurrence per year (and continent if grouped)
df_plot['Correlation_Text'] = None
group_cols = ['Year'] + (['Continent'] if plot_by == "Continent" else [])
for keys, group in df_plot.groupby(group_cols):
    first_index = group.index[0]  # Get first row index for that year & continent
    df_plot.loc[first_index, 'Correlation_Text'] = (
        f"r = {group['Correlation'].iloc[0]:.2f}" if pd.notnull(group['Correlation'].iloc[0]) else "Not enough data"
    )

# Create scatter plot
fig = px.scatter(
    df_plot,
    x=x_col,
    y=y_col,
    color="Continent" if plot_by == "Continent" else None,
    hover_name="Country",
    hover_data=["Continent"] if plot_by == "All Data" else [],
    animation_frame="Year",
    text="Correlation_Text",
    title=f"{x_col} vs {y_col} per {'Continent' if plot_by == 'Continent' else 'All Data'}"
)

if plot_by == "All Data":
    trendline_data = []
    for year in df_plot['Year'].unique():
        year_df = df_plot[df_plot['Year'] == year].dropna(subset=[x_col, y_col])
        if len(year_df) > 1:
            X = sm.add_constant(year_df[x_col])
            y_vals = year_df[y_col]
            model = sm.OLS(y_vals, X).fit()
            x_range = np.linspace(year_df[x_col].min(), year_df[x_col].max(), 100)
            y_pred = model.predict(sm.add_constant(x_range))
            trendline_data.append(pd.DataFrame({'Year': year, x_col: x_range, y_col: y_pred}))

    if trendline_data:
        trendline_df = pd.concat(trendline_data)
        fig.add_traces(px.line(trendline_df, x=x_col, y=y_col, animation_frame="Year", line_dash_sequence=["solid"])
                       .update_traces(line=dict(color="red"))
                       .data)
        
fig.update_traces(textposition="top center")

st.plotly_chart(fig)

# CORRELATION PLOT
cols_to_corr = [
    'Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)',
    'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 
    'Extreme Weather Events', 'Forest Area (%)'
]


st.subheader("Correlation Heatmap Analysis")

# Selection for category-based correlation
category_col = st.selectbox("Select a category for correlation (Optional):", ['None'] + ['Continent', 'Country'])

# Filter based on category selection
filtered_df = df.copy()

if category_col != 'None':
    unique_values = df[category_col].unique()
    selected_value = st.selectbox(f"Select a specific {category_col} (Optional):", ['All'] + list(unique_values))
    
    if selected_value != 'All':
        filtered_df = df[df[category_col] == selected_value]

# Compute correlation
corr_matrix = filtered_df[cols_to_corr].corr()

# Mask upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
plt.title(f'Correlation Heatmap ({category_col}: {selected_value if category_col != "None" else "All Data"})')

st.pyplot(fig)
# ANOVA

# Function to check normality using Shapiro-Wilk test
def check_normality(data):
    stat, p_value = stats.shapiro(data)
    return p_value > 0.05

# Function to check homoscedasticity using Levene's Test
def check_homoscedasticity(*groups):
    stat, p_value = stats.levene(*groups)
    return p_value > 0.05

# Function to perform ANOVA analysis
def anova_analysis(grouped_data, check=True):
    results = []

    for col in cols_:  
        normality_status = {}
        homoscedasticity_met = True

        # Check normality for each continent
        if check:
            for continent in grouped_data['Continent'].unique():
                data = grouped_data[grouped_data['Continent'] == continent][col]
                normality_status[continent] = check_normality(data)

        # Check homoscedasticity for all continents
        groups = [grouped_data[grouped_data['Continent'] == continent][col] for continent in grouped_data['Continent'].unique()]
        homoscedasticity_met = check_homoscedasticity(*groups)

        # Choose ANOVA or Kruskal-Wallis based on assumptions
        x = stats.f_oneway if all(normality_status.values()) and homoscedasticity_met else stats.kruskal
        f_val, p_val = x(*groups)

        # Store results
        results.append({
            'Variable': col,
            'F/K Value': f"{f_val:.2f}",
            'p-value': f"{p_val:.4f}",
            'Significant': "Yes" if p_val < 0.05 else "No",
            'Normality': "✅ Passed" if all(normality_status.values()) else "❌ Failed",
            'Homoscedasticity': "✅ Passed" if homoscedasticity_met else "❌ Failed"
        })

    return pd.DataFrame(results)

# Streamlit UI
st.subheader("ANOVA Analysis with Normality & Homoscedasticity Checks")



# Define numerical columns
cols_ = ['Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)',
         'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 
         'Extreme Weather Events', 'Forest Area (%)']

# Group by Continent and Year
grouped_data = df.groupby(['Continent', 'Year'])[cols_].mean().reset_index()

# Run ANOVA
anova_df = anova_analysis(grouped_data, check=True)

# Display results
st.write("ANOVA/Kruskal-Wallis Test Results For Continents")
st.dataframe(anova_df)

# Aggregate CO2 emissions by continent
top_6_continents = df.groupby("Continent")["CO2 Emissions (Tons/Capita)"].sum().nlargest(6)

# Create a doughnut chart (pie chart with hole in the center)
fig = px.pie(
    top_6_continents, 
    names=top_6_continents.index, 
    values=top_6_continents.values,
    title="Proportional CO2 Emissions by Continent",
    hole=0.3,  # Creates the doughnut effect
    labels={'names': 'Continent', 'values': 'CO2 Emissions (Tons/Capita)'}
)

# Update layout to add lines pointing from the circle to the legend
fig.update_traces(textinfo='percent+label', pull=[0.1] * len(top_6_continents))  
fig.update_layout(
    showlegend=True, 
    legend=dict(
        orientation="v",
        x=1,  # Position the legend
        xanchor="left",
        y=1,
        yanchor="top"
    )
)

# Display the figure in Streamlit
st.plotly_chart(fig)

# Add disclaimer about missing data
st.warning(
    "**Note:** This distribution does not accurately reflect real-world CO2 emissions. "
    "The dataset has missing values across continents (e.g., only South Africa is represented for Africa)."
)

# 
st.title("Global Distribution of Environmental Metrics")
st.write('                          \n               \n')
# Function to choose color scale based on the selected feature
def choose(selected_feature):
    if selected_feature == 'Forest Area (%)':
        return 'greens'
    elif selected_feature in ['Rainfall (mm)', 'Sea Level Rise (mm)']:
        return 'Blues'
    else:
        return 'reds'


# Sort data by Year
df_plot = df.sort_values('Year', ascending=True)
# User selects a numerical column to visualize
selected_feature = st.selectbox("Select a metric to display:", cols_, index=0)

# Create the choropleth map
fig = px.choropleth(
    df_plot,
    locations='ISO Country Code',
    color=selected_feature,
    hover_name='Continent',
    color_continuous_scale=choose(selected_feature),
    animation_frame='Year',
    title=f'#### Global Distribution of {selected_feature}'
)

# Display the figure in Streamlit
st.plotly_chart(fig)

# RANKING
def rank(df, year, feature_weights=None, num_clusters=5):
    filtered_df = df[df['Year'] == year].copy()

    # Features for ranking
    features = cols_

    data_for_clustering = filtered_df[features].copy()
    data_for_clustering = data_for_clustering.fillna(data_for_clustering.mean())

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_for_clustering)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    filtered_df.loc[:, 'Cluster'] = kmeans.fit_predict(normalized_data)

    # Assign weights (equal if not provided)
    if feature_weights is None:
        feature_weights = {feature: 1 for feature in features}

    weighted_features = [filtered_df[feature] * feature_weights[feature] for feature in features]
    filtered_df.loc[:, 'Risk Score'] = sum(weighted_features)
    filtered_df.loc[:, 'Rank'] = filtered_df['Risk Score'].rank(ascending=False, method='min')

    return filtered_df

# Function to visualize ranking on a world map
def visualize_rank(filtered_df):
    year = filtered_df['Year'].iloc[0]

    fig = px.choropleth(
        filtered_df, locations='ISO Country Code', color='Rank',
        hover_name='Country', hover_data=['Risk Score'],
        color_continuous_scale='reds',
        title=f'🌍Climate Change Risk Ranking for {year}'
    )
    fig.update_geos(showcoastlines=True, coastlinecolor="Black")
    return fig

# Year selection slider (2000 - 2023)
selected_year = st.slider("Select a Year:", min_value=2000, max_value=2023, value=2023)

# Rank countries for selected year
ranked_df = rank(df, selected_year)

# Show the global distribution map
st.plotly_chart(visualize_rank(ranked_df))

# Show ranked data in a table
st.subheader("Country Rankings")
st.dataframe(ranked_df[['Country', 'Rank', 'Risk Score']].sort_values(by="Rank").reset_index(drop=True))





