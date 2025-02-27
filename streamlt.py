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

url = "https://raw.githubusercontent.com/kenzyxriah/climate-change-analysis-2023/main/climate_change_mod.csv"

# Streamlit App Title
# Streamlit App Title
# Set page layout to wide
st.set_page_config(layout="wide")

# Title
st.title("🌍 Climate Change Impact and Risk Assessment")


col1, col2, col3, col4 = st.columns(4)

# # Display summary stats
# col1.metric("Year Range", "2000 - 2023")
# col2.metric("Continents", "6")
# col3.metric("Countries", "25")
# col4.metric("Data Points", "1000+")



# Custom metric display using markdown for full control
col1.markdown('<p class="metric-label">Year Range</p><p class="metric-value">2000 - 2023</p>', unsafe_allow_html=True)
col2.markdown('<p class="metric-label">Continents</p><p class="metric-value">6</p>', unsafe_allow_html=True)
col3.markdown('<p class="metric-label">Countries</p><p class="metric-value">25</p>', unsafe_allow_html=True)
col4.markdown('<p class="metric-label">Data Points</p><p class="metric-value">1000+</p>', unsafe_allow_html=True)

# Inject CSS for styling
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }  /* Reduce padding at the top */
    .metric-label { 
        font-size: 20px !important;  /* Smaller labels */
        font-weight: bold;
        color: #555;
    }
    .metric-value { 
        font-size: 15px !important;  /* Smaller numbers */
        font-weight: normal;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# # Introduction Section
# st.markdown("""
# ## 📌 Project Overview  
# Climate change is one of the most pressing global challenges, affecting ecosystems, economies, and human livelihoods. Rising temperatures, increasing CO₂ emissions, deforestation, and extreme weather events are contributing to a complex web of environmental issues.  
# This project provides a **data-driven approach** to understanding climate change’s impact across different continents and countries using **interactive visualizations and statistical analysis**.  

# ## 🔍 Why This Analysis?  
# The purpose of this analysis is to:  

# 1️⃣ **Track Global Climate Trends** 📊  
#    - Analyze how key climate variables (e.g., temperature, CO₂ emissions, rainfall, sea level rise) have changed over time.  
#    - Provide a **visual understanding** of climate patterns and their **regional variations**.  

# 2️⃣ **Assess Climate-Related Risks** ⚠️  
#    - Rank countries based on climate impact indicators using a **composite risk score**.  
#    - Identify regions that are most vulnerable to climate change effects.  

# 3️⃣ **Understand Relationships Between Climate Variables** 🔬  
#    - Explore correlations between **CO₂ emissions, temperature, deforestation, and extreme weather events**.  
#    - Perform statistical tests (ANOVA) to examine differences across continents.  

# 4️⃣ **Enable Interactive Data Exploration** 🎛️  
#    - Select and filter data dynamically using Streamlit’s interactive widgets.  
#    - Compare climate trends across different continents and timeframes.  

# By providing these insights, this project aims to **enhance awareness** and encourage **data-driven climate action**. 🌱🌎  
# """)

# # -----------------------------------------------------------------------------
# Declare some useful functions.
@st.cache_data
def load_data():
    return pd.read_csv(url)  

df = load_data()

df['Year'] = df['Year'].astype(str)
# Define columns
cat_cols = ['Year', 'Country', 'ISO Country Code', 'Continent']
numeric_cols = ['Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 
                'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 'Extreme Weather Events',
                'Forest Area (%)'] 

# Create a dropdown expander for dataset
with st.expander("📊 View Dataset", expanded=False):
    st.dataframe(df[cat_cols + numeric_cols])
    

colA, colB = st.columns([0.3, 0.3])  # Two equal columns

with colA:
    st.markdown('KDE Distribution Plot')

    # Create columns for side-by-side selection
    col_a1, col_a2, col_a3 = st.columns([0.4, 0.4, 0.4])  # Adjust ratio for spacing

    with col_a1:
        x = st.selectbox("X-axis:", numeric_cols, key="x_axis")

    with col_a2:
        y = st.selectbox("Group by:", ['Year', 'Continent'], key="group_by")

    with col_a3:
        include_hist = st.checkbox("Histogram", value=True)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3.2))
    sns.kdeplot(data=df, x=x, hue=y, fill=False, common_norm=False, ax=ax)

    if include_hist:
        ax1 = ax.twinx()
        sns.histplot(data=df, x=x, hue=y, element="step", common_norm=False, alpha=0.3, ax=ax1)

    ax.set_xlabel(x)
    ax.set_ylabel("Density")

    st.pyplot(fig)
    
dfa = df.copy()
dfa['Year'] = dfa['Year'].astype(int)
with colB:
    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        bar_x = st.selectbox("Select the numerical column for bar plot:", numeric_cols, key='bar_x')
    
    with col_b2:
        stack_by_continent = st.checkbox("Stack by Continent", value=False, key='stack_continent')

    colors = ['#857650', '#498582', '#3f4543', '#54657d', '#855a4e', '#b83b25']


    # Year Range Slider
    year_range = st.slider("", int(dfa['Year'].min()), int(dfa['Year'].max()), (int(dfa['Year'].min()), int(dfa['Year'].max())))
    # Hide the labels under the slider using custom CSS
    st.markdown(
    """
    <style>
    label[data-testid="stWidgetLabel"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    df_filtered = dfa[(dfa['Year'] >= year_range[0]) & (dfa['Year'] <= year_range[1])]

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    if stack_by_continent:
        sns.barplot(data=df_filtered, x='Year', y=bar_x, hue='Continent', ax=ax4, dodge=False, palette = colors, errorbar = None)
        plt.legend(title='Category')
        # Display legend at the bottom
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=6)
        
    else:
        sns.barplot(data=df_filtered, x='Year', y=bar_x, ax=ax4, color='#607cb5', errorbar = None)

    ax4.set_title(f"Bar Plot of {bar_x} Over Time")
    ax4.set_xlabel("Year")
    ax4.set_ylabel(bar_x)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)

    st.pyplot(fig4)
    

# SCATTER PLOT AND HEATMAP
st.subheader("Correlations and Distribution")
df['Year'] = df['Year'].astype(int)

cat_cols = ['Year', 'Country', 'ISO Country Code', 'Continent']
numeric_cols = ['Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 
                'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 'Extreme Weather Events',
                'Forest Area (%)']

# Create two columns: A | B Plots Stacked in left column, C | D stacked in right column (merged)
col1, col2 = st.columns([0.5, 0.9])  # Adjust width ratio as needed

# Left Column: Subplots (Scatter/Heatmap + Doughnut Chart)
with col1:
    # Dropdown for switching between Scatter Plot and Correlation Heatmap
    plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Correlation Heatmap"], key="switch_plot")

    # Define subplot for col1 (fig_a)
    fig_a, ax_a = plt.subplots()

    if plot_type == "Scatter Plot":
        # Scatter Plot Selection
        col1a, col1b, col1c = st.columns([0.2, 0.2, 0.2])

        with col1a:
            x_col = st.selectbox("Select X-axis (Numerical):", numeric_cols, key="X")
        with col1b:
            y_col = st.selectbox("Select Y-axis (Numerical):", numeric_cols, key="scatter_y")
        with col1c:
            plot_by = st.radio("Plot By:", ["Continent", "All Data"], key="scatter_plot_by", horizontal=True)

        # Sort data by Year
        df_plot = df.sort_values(by="Year", ascending=True)

        # Compute correlation per continent per year
        if plot_by == "Continent":
            correlation_per_group = df_plot.groupby(["Year", "Continent"]).apply(
                lambda x: x[[x_col, y_col]].corr().iloc[0, 1] if len(x) > 1 else None
            ).reset_index(name="Correlation")
        else:
            correlation_per_group = df_plot.groupby(["Year"]).apply(
                lambda x: x[[x_col, y_col]].corr().iloc[0, 1] if len(x) > 1 else None
            ).reset_index(name="Correlation")

        # Merge correlation data
        df_plot = df_plot.merge(
            correlation_per_group,
            on=["Year"] + (["Continent"] if plot_by == "Continent" else []),
            how="left"
        )

        # Assign correlation text only to the first occurrence per year (and continent if grouped)
        df_plot["Correlation_Text"] = None
        group_cols = ["Year"] + (["Continent"] if plot_by == "Continent" else [])
        for keys, group in df_plot.groupby(group_cols):
            first_index = group.index[0]  # Get first row index for that year & continent
            df_plot.loc[first_index, "Correlation_Text"] = (
                f"r = {group['Correlation'].iloc[0]:.2f}" if pd.notnull(group["Correlation"].iloc[0]) else "Not enough data"
            )

        # Create scatter plot
        fig_a = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            color="Continent" if plot_by == "Continent" else None,
            hover_name="Country",
            hover_data=["Continent"] if plot_by == "All Data" else [],
            animation_frame="Year",
            text="Correlation_Text",trendline = 'ols'if plot_by == "All Data" else None, 
            title=f"{x_col} vs {y_col} per {'Continent' if plot_by == 'Continent' else 'All Data'}"
        )

        fig_a.update_traces(textposition="top center")
        st.plotly_chart(fig_a)

    else:
        # Correlation Heatmap
        cols_to_corr = [
            "Avg Temperature (°C)", "CO2 Emissions (Tons/Capita)", "Sea Level Rise (mm)",
            "Rainfall (mm)", "Population", "Renewable Energy (%)",
            "Extreme Weather Events", "Forest Area (%)"
        ]
        col1a, col1b, col1c = st.columns([0.2, 0.2, 0.2])

        # Selection for category-based correlation
        with col1a: 
            category_col = st.selectbox("Select a category for correlation (Optional):", ["None"] + ["Continent", "Country"])

        # Filter based on category selection
        filtered_df = df.copy()

        if category_col != "None":
            unique_values = df[category_col].unique()
            with col1b:
                selected_value = st.selectbox(f"Select a specific {category_col} (Optional):", ["All"] + list(unique_values))

            if selected_value != "All":
                filtered_df = df[df[category_col] == selected_value]

        # Compute correlation
        corr_matrix = filtered_df[cols_to_corr].corr()

        # Mask upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot heatmap
        fig_a, ax_a = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
        plt.title(f"Correlation Heatmap ({category_col}: {selected_value if category_col != 'None' else 'All Data'})")

        st.pyplot(fig_a)

    # **CO₂ Emissions Doughnut Chart (fig_b)**
    # This remains in col1 as a **subplot** under fig_a
    top_6_continents = dfa.groupby("Continent")["CO2 Emissions (Tons/Capita)"].sum().nlargest(6)

    fig_b = px.pie(
        top_6_continents,
        names=top_6_continents.index,
        values=top_6_continents.values,
        title="Proportional CO2 Emissions by Continent",
        hole=0.3,  # Creates the doughnut effect
        labels={"names": "Continent", "values": "CO2 Emissions (Tons/Capita)"}
    )

    fig_b.update_traces(textinfo="percent+label", pull=[0.1] * len(top_6_continents))
    fig_b.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1,  # Position the legend
            xanchor="left",
            y=1,
            yanchor="top"
        )
    )

    # Display fig_b (CO₂ Emissions Doughnut) **below fig_a**
    st.plotly_chart(fig_b)

    # Add disclaimer about missing data
    st.warning(
        "**Note:** This distribution does not accurately reflect real-world CO2 emissions. "
        "The dataset has missing values across continents (e.g., only South Africa is represented for Africa)."
    )


# RANKING
def rank(df, year, feature_weights=None, num_clusters=5):
    filtered_df = df[df['Year'] == year].copy()

    # Features for ranking
    features = [
            "Avg Temperature (°C)", "CO2 Emissions (Tons/Capita)", "Sea Level Rise (mm)",
            "Rainfall (mm)", "Population", "Renewable Energy (%)",
            "Extreme Weather Events", "Forest Area (%)"
        ]

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
        title=f'🌍Climate Change Risk Ranking for {year}',
    )
    fig.update_geos(showcoastlines=True, coastlinecolor="Black")
    return fig


with col2:
    # LINE TREND
    st.subheader('Line Trend Analysis')
    figa, ax_a = plt.subplots()
    col2a, col2b = st.columns([0.5, 0.5])

    # Place widgets in col1
    with col2a:
        trend_x = st.selectbox("Select the numerical column for trend analysis:", numeric_cols, key='trend_x')

    with col2b:
        include_projection = st.checkbox("Include Projection Trend", value=False, key='projection')

    # Group data by Year and calculate percentage change

    year = dfa.groupby('Year', as_index=False)[trend_x].mean()
    year['pct_change'] = year[trend_x].pct_change() * 100

    # Create the plot
    fig2, ax2 = plt.subplots(figsize=(12, 3.3))

    sns.lineplot(x='Year', y=trend_x, data=year, ax=ax2, color='k', marker='o', linestyle='--', markerfacecolor='red', errorbar=None)

    # Annotate percentage change
    for i, value in enumerate(year['pct_change']):
        if pd.notna(value):  
            ax2.text(year['Year'].iloc[i], year[trend_x].iloc[i], f'{value:.2f}%', 
                    ha='center', color='black', fontsize=10)

    plt.axvspan(2013, 2023, color="orange", alpha=0.3, label="Last Decade")
    ax2.set_xlabel('Year')
    ax2.set_ylabel(trend_x)

    # Include projection if checkbox is checked
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
        
        if accuracy >= 85:
            
            st.write(f"Mean Absolute Error (MAE) on training data: {mae:.2f} |  Model Accuracy: {accuracy:.2f}% |  The model has reached a minimum accuracy of 85%")
        else:
            st.write(f"Mean Absolute Error (MAE) on training data: {mae:.2f} |  Model Accuracy: {accuracy:.2f}% |  The model has not reached a minimum accuracy of 85%")

    st.pyplot(fig2)

    # RANKED COUNTRY
    st.subheader('Climate Risk Ranking')
    fig, ax = plt.subplots()
    selected_year = st.slider("Select a Year:", min_value=2000, max_value=2023, value=2023)


    # Rank countries for selected year
    ranked_df = rank(df, selected_year)

    # Show the global distribution map
    st.plotly_chart(visualize_rank(ranked_df))
    with st.expander("📊 View Dataset", expanded=False):
        st.dataframe(ranked_df[['Country', 'Rank', 'Risk Score']].sort_values(by="Rank", ascending = False).reset_index(drop=True))

# # ANOVA

# # Function to check normality using Shapiro-Wilk test
# def check_normality(data):
#     stat, p_value = stats.shapiro(data)
#     return p_value > 0.05

# # Function to check homoscedasticity using Levene's Test
# def check_homoscedasticity(*groups):
#     stat, p_value = stats.levene(*groups)
#     return p_value > 0.05

# # Function to perform ANOVA analysis
# def anova_analysis(grouped_data, check=True):
#     results = []

#     for col in cols_:  
#         normality_status = {}
#         homoscedasticity_met = True

#         # Check normality for each continent
#         if check:
#             for continent in grouped_data['Continent'].unique():
#                 data = grouped_data[grouped_data['Continent'] == continent][col]
#                 normality_status[continent] = check_normality(data)

#         # Check homoscedasticity for all continents
#         groups = [grouped_data[grouped_data['Continent'] == continent][col] for continent in grouped_data['Continent'].unique()]
#         homoscedasticity_met = check_homoscedasticity(*groups)

#         # Choose ANOVA or Kruskal-Wallis based on assumptions
#         x = stats.f_oneway if all(normality_status.values()) and homoscedasticity_met else stats.kruskal
#         f_val, p_val = x(*groups)

#         # Store results
#         results.append({
#             'Variable': col,
#             'F/K Value': f"{f_val:.2f}",
#             'p-value': f"{p_val:.4f}",
#             'Significant': "Yes" if p_val < 0.05 else "No",
#             'Normality': "✅ Passed" if all(normality_status.values()) else "❌ Failed",
#             'Homoscedasticity': "✅ Passed" if homoscedasticity_met else "❌ Failed"
#         })

#     return pd.DataFrame(results)

# # Streamlit UI
# st.subheader("ANOVA Analysis with Normality & Homoscedasticity Checks")



# # Define numerical columns
# cols_ = ['Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)',
#          'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 
#          'Extreme Weather Events', 'Forest Area (%)']

# # Group by Continent and Year
# grouped_data = df.groupby(['Continent', 'Year'])[cols_].mean().reset_index()

# # Run ANOVA
# anova_df = anova_analysis(grouped_data, check=True)

# # Display results
# st.write("ANOVA/Kruskal-Wallis Test Results For Continents")
# st.dataframe(anova_df)






