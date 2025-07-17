from typing import List
from datetime import datetime

# Outline for a natural gas storage contract pricer

def price_nat_gas_contract(
    injection_dates: List[str],  # Dates when gas is purchased (YYYY-MM-DD)
    injection_prices: List[float],  # Prices for each date when gas is purchased (cost per volume)
    withdrawal_dates: List[str],  # Dates when gas is sold (YYYY-MM-DD)
    withdrawal_prices: List[float],  # Prices for each date when gas is sold (cost per volume)
    pump_rate: float,  # Max rate at which gas can be pumped per day (volume / day)
    max_volume: float,  # Maximum storage volume (volume)
    storage_costs: float,  # Storage cost per day (cost per volume)
) -> float:
    """
    Calculate the value of a natural gas storage contract.

    We will use the following assumptions:
    - We will not consider weekends, holidays, or bank holidays
    - We will not consider transport delays
    - We will not consider interest rates
    - The injection and withdrawal dates line up with the prices on said dates
    - Injection and withdrawal dates don't overlap
    
    Args:
        injection_dates: Dates when gas is purchased (YYYY-MM-DD)
        injection_prices: Prices for each date when gas is purchased (cost per volume)
        withdrawal_dates: Dates when gas is sold (YYYY-MM-DD)
        withdrawal_prices: Prices for each date when gas is sold (cost per volume)
        pump_rate: The volume of gas that can be pumped per day (volume / day)
        max_volume: Maximum storage volume (volume)
        storage_costs: Storage cost per day (cost per volume per day)
    Returns:
        The value of the contract as a float.
    """
    # 1. Initialize storage state and value
    contract_value = 0.0
    current_volume = 0.0

    # 2. Sort the dates and prices pairs in ascending order
    # Sort injection dates and prices
    injection_pairs = list(zip(injection_dates, injection_prices))
    injection_pairs.sort(key=lambda x: x[0])  # Sort by date
    injection_dates, injection_prices = zip(*injection_pairs)
    
    # Sort withdrawal dates and prices
    withdrawal_pairs = list(zip(withdrawal_dates, withdrawal_prices))
    withdrawal_pairs.sort(key=lambda x: x[0])  # Sort by date
    withdrawal_dates, withdrawal_prices = zip(*withdrawal_pairs)

    # 3. Make an list of all the dates from the first injection date to the last withdrawal date
    # Combine all dates and create a data structure with date, injection status, and price
    all_dates = []
    
    # Add injection dates
    for date, price in zip(injection_dates, injection_prices):
        all_dates.append({
            'date': date,
            'is_injection': True,
            'price': price
        })
    
    # Add withdrawal dates
    for date, price in zip(withdrawal_dates, withdrawal_prices):
        all_dates.append({
            'date': date,
            'is_injection': False,
            'price': price
        })
    
    # Sort by date
    all_dates.sort(key=lambda x: x['date'])
    
    # Debug: Print the all_dates data structure
    print("All dates data structure:")
    for entry in all_dates:
        action = "INJECT" if entry['is_injection'] else "WITHDRAW"
        print(f"  {entry['date']}: {action} at price {entry['price']}")

    # 4. Loop over all the dates and calculate the contract value

    last_date = all_dates[0]['date']

    for date in all_dates:
        # Subtract the storage costs since the last date
        contract_value -= storage_costs * current_volume * (datetime.strptime(date['date'], '%Y-%m-%d') - datetime.strptime(last_date, '%Y-%m-%d')).days

        if date['is_injection'] and current_volume + pump_rate <= max_volume:  # If the date is an injection and the projected volume is less than the max volume, inject the gas
            current_volume += pump_rate
            contract_value -= date['price'] * pump_rate
        elif not date['is_injection'] and current_volume - pump_rate >= 0:  # If the date is a withdrawal and the projected volume is greater than 0, withdraw the gas
            current_volume -= pump_rate
            contract_value += date['price'] * pump_rate
        
        last_date = date['date']
    
    return contract_value
