import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_data(file_path) -> pd.DataFrame:
    """
    Load the data from the CSV file and display basic information.
    """
    # Load the data from the CSV file
    df = pd.read_csv(file_path)
    
    # Display basic information
    display_basic_information(df)
    return df

def display_basic_information(df) -> None:
    """
    Display basic information about the dataframe.
    """
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:")
    print(df.head())

def plot_data(df) -> None:
    """
    Plot the natural gas price data.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(df['Dates'], df['Prices'], marker='o', linewidth=2, markersize=6, color='#2E86AB', alpha=0.8, label='Natural Gas Prices')
    plt.title('Natural Gas Price Trends Over Time', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Make x-axis labels less frequent
    ax = plt.gca()
    # Show fewer x-axis ticks (adjust the number as needed)
    ax.xaxis.set_major_locator(MaxNLocator(8))  # Show max 8 ticks on x-axis
    
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Price', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()

def plot_model_predictions(y_pred_test, y_pred_train, nat_gas_df, X_train, X_test) -> None:
    """
    Plot actual vs predicted values and the original data with predictions.
    
    Parameters:
    -----------
    y_pred_test : array-like
        Predicted values for test set
    y_pred_train : array-like
        Predicted values for training set
    nat_gas_df : pd.DataFrame
        Original natural gas dataframe with dates as index
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Original data with predictions
    ax1.plot(nat_gas_df.index, nat_gas_df['Prices'], 'o-', color='#2E86AB', 
             linewidth=2, markersize=6, alpha=0.8, label='Actual Prices')
    
    # Plot training predictions
    train_dates = nat_gas_df.index[nat_gas_df.index.isin(X_train.index)]
    ax1.plot(train_dates, y_pred_train, 's', color='#28A745', 
             markersize=8, alpha=0.8, label='Training Predictions')
    
    # Plot test predictions
    test_dates = nat_gas_df.index[nat_gas_df.index.isin(X_test.index)]
    ax1.plot(test_dates, y_pred_test, '^', color='#DC3545', 
             markersize=8, alpha=0.8, label='Test Predictions')
    
    ax1.set_title('Natural Gas Prices: Actual vs Predicted', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Scatter plot of actual vs predicted
    # Get actual values for train and test sets
    y_train_actual = nat_gas_df.loc[X_train.index, 'Prices']
    y_test_actual = nat_gas_df.loc[X_test.index, 'Prices']
    
    # Perfect prediction line
    min_val = min(min(y_train_actual), min(y_test_actual), min(y_pred_train), min(y_pred_test))
    max_val = max(max(y_train_actual), max(y_test_actual), max(y_pred_train), max(y_pred_test))
    
    ax2.scatter(y_train_actual, y_pred_train, color='#28A745', alpha=0.7, 
                s=60, label='Training Set')
    ax2.scatter(y_test_actual, y_pred_test, color='#DC3545', alpha=0.7, 
                s=60, label='Test Set')
    
    # Add perfect prediction line
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
    
    ax2.set_title('Actual vs Predicted Values', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Actual Prices', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Prices', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

def feature_engineering(df) -> pd.DataFrame:
    """
    Perform feature engineering on the natural gas price data.
    """

    #Convert the Dates column to a datetime object
    df['Dates'] = pd.to_datetime(df['Dates'])

    #Set the Dates column as the index
    df.set_index('Dates', inplace=True)

    # Create additional time features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    df['Quarter'] = df.index.quarter

    # Display basic information
    display_basic_information(df)

    return df

def predict_future_months(model, nat_gas_df, months_ahead=12) -> pd.DataFrame:
    """
    Generate predictions for the next N months after the last date in the dataset.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to make predictions
    nat_gas_df : pd.DataFrame
        Original natural gas dataframe with dates as index
    months_ahead : int
        Number of months to predict into the future (default: 12)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with future dates and predictions
    """
    # Get the last date from the original dataset
    last_date = nat_gas_df.index.max()
    
    # Generate future dates (next N months)
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                periods=months_ahead, 
                                freq='M')
    
    # Create a DataFrame with future dates
    future_df = pd.DataFrame({'Dates': future_dates})
    future_df.set_index('Dates', inplace=True)
    
    # Apply feature engineering to get time features
    future_df['Year'] = future_df.index.year
    future_df['Month'] = future_df.index.month
    future_df['Day'] = future_df.index.day
    future_df['DayOfWeek'] = future_df.index.dayofweek
    future_df['DayOfYear'] = future_df.index.dayofyear
    future_df['Quarter'] = future_df.index.quarter
    
    # Select the same features used for training
    feature_columns = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Quarter']
    X_future = future_df[feature_columns]
    
    # Make predictions
    future_predictions = model.predict(X_future)
    
    # Add predictions to the dataframe
    future_df['Predicted_Prices'] = future_predictions
    
    print(f"Generated predictions for {months_ahead} months starting from {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"Future predictions shape: {future_df.shape}")
    
    return future_df

def plot_future_predictions(nat_gas_df, future_df) -> None:
    """
    Plot the original data along with future predictions.
    
    Parameters:
    -----------
    nat_gas_df : pd.DataFrame
        Original natural gas dataframe with dates as index
    future_df : pd.DataFrame
        Future predictions dataframe
    """
    plt.figure(figsize=(14, 8))
    
    # Plot original data
    plt.plot(nat_gas_df.index, nat_gas_df['Prices'], 'o-', 
             color='#2E86AB', linewidth=2, markersize=6, alpha=0.8, 
             label='Historical Prices')
    
    # Plot future predictions
    plt.plot(future_df.index, future_df['Predicted_Prices'], 's-', 
             color='#DC3545', linewidth=2, markersize=8, alpha=0.8, 
             label='Future Predictions')
    
    plt.title('Natural Gas Prices: Historical Data and Future Predictions', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Price', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()