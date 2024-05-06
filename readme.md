# DNN Lab - Financial Forecasting Application

## Overview
DNN Lab is an advanced financial analysis tool powered by deep neural networks, designed to offer predictive insights into stock market trends. This Streamlit-based web application integrates cutting-edge AI to process and visualize financial data, helping traders and analysts make informed decisions.

## Problem
Financial markets are volatile and complex, making it challenging for traders and analysts to predict future stock prices accurately. Traditional statistical methods often fall short in capturing the non-linear patterns in financial time series data, leading to suboptimal trading strategies.

## Solution
DNN Lab addresses these challenges by utilizing sophisticated deep learning models that can learn from large amounts of historical financial data and identify underlying patterns more effectively than traditional models. The application provides:
- **Real-time Predictions**: Generates forecasts for future stock prices using data from Yahoo Finance.
- **Interactive Visualizations**: Offers dynamic graph visualizations of the stock data and predictions using Plotly, enhancing user understanding and interaction.
- **AI-Enhanced Stock Lookup**: Leverages OpenAI's GPT model to interpret user queries and fetch relevant stock symbols, streamlining the data retrieval process.
- **Customizable Analysis**: Users can input different stocks and adjust parameters to see how changes might affect predictions, allowing for tailored analysis.

## Technologies
DNN Lab is built using several cutting-edge technologies and libraries:
- **Streamlit**: For creating the web application interface, providing a smooth and interactive user experience.
- **PyTorch**: Utilized for building and training the deep learning models, thanks to its flexibility and efficiency with complex neural network architectures.
- **Plotly**: Used for creating interactive and dynamic charts that help in visualizing the stock data and the model's predictions.
- **Yahoo Finance (yfinance)**: Provides real-time stock prices and historical financial data, essential for the model's training and prediction phases.
- **OpenAI GPT**: Processes natural language inputs to enhance user interaction and streamline the data retrieval process.
- **Python-dotenv**: Manages environment variables, ensuring sensitive information such as API keys are kept secure.

## Running the Application
To run DNN Lab locally, follow these steps:
1. **Set up your Python environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # Unix/macOS
    env\Scripts\activate  # Windows
    ```
2. **Install dependencies**:
    ```bash
    pip install streamlit torch numpy pandas plotly sklearn yfinance python-dotenv
    ```
4. **Create .env file and a variable called OPENAI_API_KEY **:
    ```bash
    OPENAI_API_KEY = 'your key'
    ```
3. **Launch the application**:
    ```bash
    streamlit run DNN_forecast.py
    ```

## Future Enhancements
- **Model Optimization**: Continuous improvement of the neural network models to enhance accuracy and reduce computational costs.
- **Expansion of Data Sources**: Integration of additional data sources like social media sentiment and macroeconomic indicators to improve prediction capabilities.
- **User Account System**: Implementation of a user system to save preferences and historical analysis.
