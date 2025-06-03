import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
from datetime import datetime, timedelta
from scipy import stats

# Streamlit app configuration
st.set_page_config(page_title="Quantitative Research Tool", layout="wide")

# Title
st.title("Quantitative Research Tool for Analysts")

# Sidebar for navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Data Mapping", "Data Input", "Data Analysis", "Portfolio Analysis", "Web Search"])

# Function to fetch web search results (placeholder for a search API)
def web_search(query):
    try:
        # Placeholder: Replace with your preferred search API (e.g., Google Custom Search)
        api_key = "YOUR_API_KEY"  # Replace with actual API key
        cx = "YOUR_CX"  # Replace with actual custom search engine ID
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
        response = requests.get(url)
        results = response.json().get("items", [])
        return [{"title": item["title"], "link": item["link"], "snippet": item["snippet"]} for item in results]
    except Exception as e:
        return [{"title": "Error", "link": "#", "snippet": f"Web search failed: {str(e)}. Please check API setup or disable this feature."}]

# Function to calculate VaR and CVaR
def calculate_var_cvar(returns, confidence_level=0.95):
    var = np.percentile(returns, 100 * (1 - confidence_level))
    cvar = returns[returns <= var].mean()
    return var, cvar

# Data Mapping Section
if app_mode == "Data Mapping":
    st.header("Data Mapping")
    st.write("""
    This section explains the expected data structure for uploading or entering data in the 'Data Input' section. 
    The tool is designed to work with financial or quantitative datasets, such as asset prices, returns, or other metrics.
    """)

    st.subheader("Expected Data Format")
    st.markdown("""
    - **File Type**: CSV file or manual input in CSV format (comma-separated values).
    - **Columns**:
      - **Date** (optional): Dates in `YYYY-MM-DD` format (e.g., `2023-01-01`). Used for time series analysis.
      - **Price** (optional): Numeric values representing asset prices (e.g., stock prices).
      - **Return** (optional): Numeric values representing asset returns (e.g., daily log returns).
      - **Benchmark** (optional): Returns of a benchmark index (e.g., S&P 500) for beta calculation.
      - Other numeric columns (e.g., `Volume`, `Volatility`) can be included for analysis.
    - **Data Types**:
      - `Date`: Datetime (parsed automatically from `YYYY-MM-DD`).
      - `Price`, `Return`, `Benchmark`, etc.: Float or integer.
    - **Example CSV Format**:
      ```
      Date,Price,Return,Benchmark
      2023-01-01,100.50,0.01,0.008
      2023-01-02,101.00,0.005,0.006
      2023-01-03,99.75,-0.012,-0.01
      ```
    """)

    # Sample dataset
    st.subheader("Sample Dataset")
    sample_data = pd.DataFrame({
        "Date": [datetime(2023, 1, i).strftime("%Y-%m-%d") for i in range(1, 6)],
        "Price": [100.50, 101.00, 99.75, 102.25, 103.10],
        "Return": [0.01, 0.005, -0.012, 0.025, 0.008],
        "Benchmark": [0.008, 0.006, -0.01, 0.02, 0.007]
    })
    st.dataframe(sample_data)

    # Download sample dataset as CSV
    st.subheader("Download Sample CSV")
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_quant_data.csv",
        mime="text/csv"
    )

    st.markdown("""
    **Tips for Data Upload**:
    - Ensure column names are clear and consistent (e.g., `Date`, `Price`, `Return`, `Benchmark`).
    - Use numeric values for analysis columns to enable statistical calculations and visualizations.
    - For manual input, enter data in CSV format (e.g., `Date,Price,Return,Benchmark\n2023-01-01,100.50,0.01,0.008`).
    """)

# Data Input Section
elif app_mode == "Data Input":
    st.header("Data Input")
    st.write("Upload a CSV file or enter data manually. Refer to the 'Data Mapping' tab for the expected format.")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            st.write("Data Preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    # Manual data input
    st.subheader("Manual Data Entry")
    manual_data = st.text_area("Enter data (comma-separated rows, e.g., Date,Price,Return,Benchmark):")
    if manual_data:
        try:
            df_manual = pd.read_csv(io.StringIO(manual_data))
            st.session_state['data'] = df_manual
            st.write("Manual Data Preview:")
            st.dataframe(df_manual.head())
        except Exception as e:
            st.error(f"Error parsing manual data: {str(e)}")

# Data Analysis Section
elif app_mode == "Data Analysis":
    st.header("Data Analysis")
    if 'data' not in st.session_state:
        st.warning("No data available. Please upload or enter data in the Data Input section.")
    else:
        df = st.session_state['data']
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Select columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            st.error("No numeric columns available for analysis.")
        else:
            selected_cols = st.multiselect("Select columns for analysis", numeric_cols, default=list(numeric_cols)[:2])

            if selected_cols:
                # Descriptive Statistics
                st.subheader("Descriptive Statistics")
                stats_df = df[selected_cols].describe()
                stats_df.loc['Skewness'] = df[selected_cols].skew()
                stats_df.loc['Kurtosis'] = df[selected_cols].kurtosis()
                st.write(stats_df)

                # Correlation and Covariance
                st.subheader("Correlation and Covariance")
                st.write("Correlation Matrix:")
                st.write(df[selected_cols].corr())
                st.write("Covariance Matrix:")
                st.write(df[selected_cols].cov())

                # Time Series Analysis (Moving Averages)
                if 'Date' in df.columns:
                    st.subheader("Time Series Analysis")
                    window = st.slider("Select moving average window (days)", 1, 30, 5)
                    for col in selected_cols:
                        df[f"{col}_MA"] = df[col].rolling(window=window).mean()
                        df[f"{col}_EMA"] = df[col].ewm(span=window, adjust=False).mean()
                        fig, ax = plt.subplots()
                        ax.plot(df['Date'], df[col], label=col)
                        ax.plot(df['Date'], df[f"{col}_MA"], label=f"{col} MA ({window}-day)")
                        ax.plot(df['Date'], df[f"{col}_EMA"], label=f"{col} EMA ({window}-day)")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Value")
                        ax.legend()
                        st.pyplot(fig)

                # Regression Analysis
                st.subheader("Linear Regression")
                if len(selected_cols) >= 2:
                    x_col = selected_cols[0]
                    y_col = selected_cols[1]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col].dropna(), df[y_col].dropna())
                    st.write(f"Slope: {slope:.4f}")
                    st.write(f"Intercept: {intercept:.4f}")
                    st.write(f"R-squared: {r_value**2:.4f}")
                    st.write(f"P-value: {p_value:.4f}")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                    x = df[x_col].dropna()
                    ax.plot(x, intercept + slope * x, color='red', label='Regression Line')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.legend()
                    st.pyplot(fig)

                # Visualization options
                st.subheader("Visualizations")
                plot_type = st.selectbox("Select plot type", ["Time Series", "Histogram", "Scatter Plot", "Correlation Heatmap"])

                if plot_type == "Time Series":
                    if 'Date' in df.columns:
                        try:
                            df['Date'] = pd.to_datetime(df['Date'])
                            fig, ax = plt.subplots()
                            for col in selected_cols:
                                ax.plot(df['Date'], df[col], label=col)
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Value")
                            ax.legend()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error plotting time series: {str(e)}")
                    else:
                        st.error("No 'Date' column found for time series plotting.")

                elif plot_type == "Histogram":
                    for col in selected_cols:
                        fig, ax = plt.subplots()
                        sns.histplot(df[col].dropna(), kde=True, ax=ax)
                        ax.set_title(f"Histogram of {col}")
                        st.pyplot(fig)

                elif plot_type == "Scatter Plot":
                    if len(selected_cols) >= 2:
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=df[selected_cols[0]], y=df[selected_cols[1]], ax=ax)
                        ax.set_xlabel(selected_cols[0])
                        ax.set_ylabel(selected_cols[1])
                        st.pyplot(fig)
                    else:
                        st.error("Select at least two columns for scatter plot.")

                elif plot_type == "Correlation Heatmap":
                    fig, ax = plt.subplots()
                    sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)

# Portfolio Analysis Section
elif app_mode == "Portfolio Analysis":
    st.header("Portfolio Analysis")
    if 'data' not in st.session_state:
        st.warning("No data available. Please upload or enter data in the Data Input section.")
    else:
        df = st.session_state['data']
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Select columns for portfolio analysis (e.g., asset returns)
        return_cols = st.multiselect("Select columns for portfolio returns", df.select_dtypes(include=[np.number]).columns)
        if return_cols:
            st.subheader("Portfolio Metrics")
            weights = st.slider("Select weight for first asset (remainder split equally)", 0.0, 1.0, 0.5)
            weights = [weights] + [(1 - weights) / (len(return_cols) - 1)] * (len(return_cols) - 1) if len(return_cols) > 1 else [weights]
            risk_free_rate = st.number_input("Enter annualized risk-free rate (e.g., 0.02 for 2%)", 0.0, 0.1, 0.02)

            try:
                # Calculate portfolio returns and volatility
                portfolio_returns = (df[return_cols] * weights).sum(axis=1)
                mean_return = portfolio_returns.mean() * 252  # Annualized return (assuming daily data)
                volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility

                # Sharpe and Sortino Ratios
                sharpe_ratio = (mean_return - risk_free_rate) / volatility
                downside_returns = portfolio_returns[portfolio_returns < 0]
                downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
                sortino_ratio = (mean_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else np.nan

                # VaR and CVaR
                confidence_level = st.slider("Confidence level for VaR/CVaR", 0.9, 0.99, 0.95)
                var, cvar = calculate_var_cvar(portfolio_returns, confidence_level)

                # Beta (if benchmark is available)
                beta = np.nan
                if 'Benchmark' in df.columns:
                    cov = np.cov(portfolio_returns.dropna(), df['Benchmark'].dropna())[0, 1]
                    benchmark_var = df['Benchmark'].var()
                    beta = cov / benchmark_var if benchmark_var != 0 else np.nan

                # Monte Carlo Simulation
                st.subheader("Monte Carlo Simulation")
                num_simulations = st.number_input("Number of simulations", 100, 10000, 1000)
                num_days = st.number_input("Number of days to simulate", 1, 252, 30)
                simulated_paths = []
                for _ in range(num_simulations):
                    path = [portfolio_returns.iloc[-1]]
                    for _ in range(num_days):
                        path.append(path[-1] * (1 + np.random.normal(mean_return / 252, volatility / np.sqrt(252))))
                    simulated_paths.append(path)
                simulated_paths = pd.DataFrame(simulated_paths).T

                # Display metrics
                st.write(f"Annualized Portfolio Return: {mean_return:.4f}")
                st.write(f"Annualized Portfolio Volatility: {volatility:.4f}")
                st.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")
                st.write(f"Sortino Ratio: {sortino_ratio:.4f}")
                st.write(f"VaR ({confidence_level*100}%): {var:.4f}")
                st.write(f"CVaR ({confidence_level*100}%): {cvar:.4f}")
                if not np.isnan(beta):
                    st.write(f"Beta (vs. Benchmark): {beta:.4f}")

                # Plot portfolio returns
                fig, ax = plt.subplots()
                portfolio_returns.plot(ax=ax)
                ax.set_title("Portfolio Returns")
                st.pyplot(fig)

                # Plot Monte Carlo simulation
                fig, ax = plt.subplots()
                simulated_paths.plot(ax=ax, legend=False, alpha=0.1)
                ax.set_title("Monte Carlo Simulation of Portfolio Returns")
                ax.set_xlabel("Days")
                ax.set_ylabel("Portfolio Value")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in portfolio analysis: {str(e)}")

# Web Search Section
elif app_mode == "Web Search":
    st.header("Web Search for Contextual Research")
    query = st.text_input("Enter search query (e.g., 'stock market trends 2025')")
    if query:
        results = web_search(query)
        st.subheader("Search Results")
        for result in results[:5]:  # Limit to top 5 results
            st.write(f"**{result['title']}**")
            st.write(f"[Link]({result['link']})")
            st.write(result['snippet'])
            st.write("---")

# Footer
st.sidebar.markdown("""
Built with Streamlit for quantitative analysts.  
Upload data with columns like `Date`, `Price`, `Return`, or `Benchmark` for best results.  
Refer to the **Data Mapping** tab for guidance on data format.
""")
