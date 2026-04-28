import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# --- CONFIGURATION: Supported Commodities and Currencies ---
supported_commodities = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F"
}

exchange_rates = {
    "USD": 1.0,
    "INR": 83.5,
    "EUR": 0.92,
    "GBP": 0.79
}

def run_project():
    print("--- Commodity Market Prediction System ---")
    
    # --- 1. COMMODITY SELECTION: Get user input and validate ---
    print("Available Commodities:", list(supported_commodities.keys()))
    user_choice = input("Enter commodity name: ").strip().capitalize()

    if user_choice not in supported_commodities:
        print("Error: Invalid selection.")
        return

    # --- 2. CURRENCY SELECTION: Choose display currency ---
    print("\nAvailable Currencies:", list(exchange_rates.keys()))
    selected_currency = input("Select currency: ").upper().strip()
    rate = exchange_rates.get(selected_currency, 1.0)

    symbol = supported_commodities[user_choice]
    
    try:
        # --- 3. DATA RETRIEVAL: Fetch historical prices using yfinance ---
        print(f"\nFetching data for {user_choice}...")
        data = yf.download(symbol, period="1y", interval="1d")
        
        # Fixing column headers for compatibility
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            print("Error: No data retrieved.")
            return

        # --- 4. ACCURACY CHECK: Compare model predictions against real past data ---
        df = data.reset_index()
        df['Days'] = df.index
        
        # Splitting data: We hide the last 10 days to test the model
        train_df = df.iloc[:-10]
        test_df = df.iloc[-10:]
        
        X_train = train_df[['Days']].values
        y_train = train_df['Close'].values
        X_test = test_df[['Days']].values
        y_actual = test_df['Close'].values
        
        # Testing the model on hidden data
        temp_model = LinearRegression()
        temp_model.fit(X_train, y_train)
        y_pred = temp_model.predict(X_test)
        
        # Calculating Accuracy Percentage
        mape = mean_absolute_percentage_error(y_actual, y_pred)
        accuracy = (1 - mape) * 100
        
        print(f"\n--- Model Accuracy Check ---")
        print(f"The model predicted the last 10 days with {accuracy:.2f}% accuracy.")

        # --- 5. FORECASTING: Predicting future prices for the next 30 days ---
        final_model = LinearRegression()
        final_model.fit(df[['Days']].values, df['Close'].values)
        
        future_days = np.array([[len(df) + i] for i in range(1, 31)])
        forecasted_prices_usd = final_model.predict(future_days)

        # Apply currency conversion
        converted_forecast = forecasted_prices_usd * rate
        converted_history = df['Close'].values * rate

        print(f"\n--- 30-Day Forecast in {selected_currency} for {user_choice} ---")
        for i, price in enumerate(converted_forecast[:5], 1):
            print(f"Day {i}: {selected_currency} {price:.2f}")

        # --- 6. VISUALIZATION: Plotting the results on a graph ---
        plt.figure(figsize=(10, 5))
        plt.plot(df['Days'], converted_history, label=f"History ({selected_currency})")
        plt.plot(range(len(df), len(df) + 30), converted_forecast, 
                 label=f"Forecast ({selected_currency})", color='red', linestyle='--')
        
        plt.title(f"{user_choice} Prediction (Accuracy: {accuracy:.2f}%)")
        plt.xlabel("Timeline (Days)")
        plt.ylabel(f"Price in {selected_currency}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- 7. DOWNLOAD RESULTS: Save the forecast to a CSV file ---
        save_choice = input("\nDo you want to download results as CSV? (yes/no): ").strip().lower()
        if save_choice == 'yes':
            filename = f"{user_choice}_results_{selected_currency}.csv"
            pd.DataFrame({'Day': range(1, 31), 'Price': converted_forecast}).to_csv(filename, index=False)
            print(f"Success: File saved as {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_project()