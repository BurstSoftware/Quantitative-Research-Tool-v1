import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

# Streamlit app configuration
st.set_page_config(page_title="Quantitative Research Tool", layout="wide")

# Title
st.title("Quantitative Research Tool for Analysts")

# Sidebar for navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Data Input", "Data Analysis", "Portfolio Analysis", "Web Search"])

# Function to fetch web search results (placeholder for a search API)
def web_search(query):
    try:
        # Placeholder: Replace with your preferred search API (e.g., Google Custom Search)
        # Example: https://www.googleapis.com/customsearch/v1?key=YOUR_API_KEY&cx=YOUR_CX&q=query
        api_key = "YOUR_API_KEY"  # Replace with actual API key
        cx = "YOUR_CX"  # Replace with actual custom search engine ID
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
        response = requests.get(url)
        results = response.json().get("items", [])
        return [{"title": item["title"], "link": item["link"], "snippet": item["snippet"]} for item in results]
    except Exception as e:
        return [{"title": "Error", "link": "#", "snippet": f"Web search failed: {str(e)}. Please check API setup or disable this feature."}]

# Data Input Section
if app_mode == "Data Input":
    st.header("Data Input")
    st.write("Upload a CSV file or enter data manually. Ensure the CSV has columns like 'Date', 'Price', 'Return', or other relevant metrics.")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.write("Data Preview:")
        st.dataframe(df.head())

    # Manual data input
    st.subheader("Manual Data Entry")
    manual_data = st.text_area("Enter data (comma-separated rows, e.g., Date,Price,Return):")
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
                # Basic statistics
                st.subheader("Basic Statistics")
                st.write(df[selected_cols].describe())

                # Visualization options
                st.subheader("Visualizations")
                plot_type = st.selectbox("Select plot type", ["Time Series", "Histogram", "Scatter Plot", "Correlation Heatmap"])

                if plot_type == "Time Series":
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        fig, ax = plt.subplots()
                        for col in selected_cols:
                            ax.plot(df['Date'], df[col], label=col)
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Value")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.error("No 'Date' column found for time series plotting.")

                elif plot_type == "Histogram":
                    for col in selected_cols:
                        fig, ax = plt.subplots()
                        sns.histplot(df[col], kde=True, ax=ax)
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

            # Calculate portfolio returns and volatility
            portfolio_returns = (df[return_cols] * weights).sum(axis=1)
            mean_return = portfolio_returns.mean() * 252  # Annualized return (assuming daily data)
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility

            st.write(f"Annualized Portfolio Return: {mean_return:.4f}")
            st.write(f"Annualized Portfolio Volatility: {volatility:.4f}")

            # Plot portfolio returns
            fig, ax = plt.subplots()
            portfolio_returns.plot(ax=ax)
            ax.set_title("Portfolio Returns")
            st.pyplot(fig)

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
st.sidebar.write("Built with Streamlit for quantitative analysts. Upload data with columns like 'Date', 'Price', or 'Return' for best results.")
