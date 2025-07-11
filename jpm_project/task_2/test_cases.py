# Import the pricing function
from nat_gas_contract_pricer import price_nat_gas_contract

def test_seasonal_arbitrage():
    """Test Case 1: Simple seasonal arbitrage (buy in summer, sell in winter)"""
    print("="*60)
    print("TEST CASE 1: Simple Seasonal Arbitrage")
    print("="*60)

    injection_dates = ['2024-06-15', '2024-07-01', '2024-08-15']
    injection_prices = [2.10, 2.05, 2.15]
    withdrawal_dates = ['2024-12-01', '2024-12-15', '2025-01-15']
    withdrawal_prices = [4.50, 4.80, 5.20]
    pump_rate = 5000.0  # 5000 MMBtu per day
    max_volume = 100000.0  # 100,000 MMBtu storage capacity
    storage_costs = 0.005  # $0.005 per MMBtu per day

    print(f"Injection Period: Summer 2024")
    print(f"  Dates: {injection_dates}")
    print(f"  Prices: ${injection_prices} per MMBtu")
    print(f"\nWithdrawal Period: Winter 2024-2025")
    print(f"  Dates: {withdrawal_dates}")
    print(f"  Prices: ${withdrawal_prices} per MMBtu")
    print(f"\nOperational Parameters:")
    print(f"  Pump Rate: {pump_rate:,.0f} MMBtu/day")
    print(f"  Max Storage: {max_volume:,.0f} MMBtu")
    print(f"  Storage Cost: ${storage_costs:.3f} per MMBtu/day")
    print("\n" + "-"*60)

    contract_value = price_nat_gas_contract(
        injection_dates=injection_dates,
        injection_prices=injection_prices,
        withdrawal_dates=withdrawal_dates,
        withdrawal_prices=withdrawal_prices,
        pump_rate=pump_rate,
        max_volume=max_volume,
        storage_costs=storage_costs
    )

    print(f"\nContract Value: ${contract_value:,.2f}")
    print(f"ROI per MMBtu: ${contract_value / (pump_rate * len(injection_dates)):.2f}")
    
    return contract_value

def test_high_frequency_trading():
    """Test Case 2: High frequency trading with smaller spreads"""
    print("\n\n" + "="*60)
    print("TEST CASE 2: High Frequency Trading")
    print("="*60)

    injection_dates = ['2024-03-01', '2024-03-05', '2024-03-10', '2024-03-15']
    injection_prices = [3.20, 3.15, 3.25, 3.10]
    withdrawal_dates = ['2024-03-03', '2024-03-08', '2024-03-12', '2024-03-18']
    withdrawal_prices = [3.45, 3.40, 3.50, 3.35]
    pump_rate = 2000.0  # 2000 MMBtu per day
    max_volume = 20000.0  # 20,000 MMBtu storage capacity
    storage_costs = 0.010  # $0.01 per MMBtu per day

    print(f"Injection Period: March 2024 (Weekly)")
    print(f"  Dates: {injection_dates}")
    print(f"  Prices: ${injection_prices} per MMBtu")
    print(f"\nWithdrawal Period: March 2024 (Weekly)")
    print(f"  Dates: {withdrawal_dates}")
    print(f"  Prices: ${withdrawal_prices} per MMBtu")
    print(f"\nOperational Parameters:")
    print(f"  Pump Rate: {pump_rate:,.0f} MMBtu/day")
    print(f"  Max Storage: {max_volume:,.0f} MMBtu")
    print(f"  Storage Cost: ${storage_costs:.3f} per MMBtu/day")
    print("\n" + "-"*60)

    contract_value = price_nat_gas_contract(
        injection_dates=injection_dates,
        injection_prices=injection_prices,
        withdrawal_dates=withdrawal_dates,
        withdrawal_prices=withdrawal_prices,
        pump_rate=pump_rate,
        max_volume=max_volume,
        storage_costs=storage_costs
    )

    print(f"\nContract Value: ${contract_value:,.2f}")
    print(f"ROI per MMBtu: ${contract_value / (pump_rate * len(injection_dates)):.2f}")
    
    return contract_value

def test_unprofitable_scenario():
    """Test Case 3: Unprofitable scenario (storage costs too high)"""
    print("\n\n" + "="*60)
    print("TEST CASE 3: Unprofitable Scenario (High Storage Costs)")
    print("="*60)

    injection_dates = ['2024-05-01', '2024-05-15']
    injection_prices = [2.80, 2.75]
    withdrawal_dates = ['2024-11-01', '2024-11-15']
    withdrawal_prices = [3.20, 3.25]
    pump_rate = 3000.0  # 3000 MMBtu per day
    max_volume = 50000.0  # 50,000 MMBtu storage capacity
    storage_costs = 0.050  # $0.05 per MMBtu per day (very high)

    print(f"Injection Period: Spring 2024")
    print(f"  Dates: {injection_dates}")
    print(f"  Prices: ${injection_prices} per MMBtu")
    print(f"\nWithdrawal Period: Fall 2024")
    print(f"  Dates: {withdrawal_dates}")
    print(f"  Prices: ${withdrawal_prices} per MMBtu")
    print(f"\nOperational Parameters:")
    print(f"  Pump Rate: {pump_rate:,.0f} MMBtu/day")
    print(f"  Max Storage: {max_volume:,.0f} MMBtu")
    print(f"  Storage Cost: ${storage_costs:.3f} per MMBtu/day (HIGH)")
    print("\n" + "-"*60)

    contract_value = price_nat_gas_contract(
        injection_dates=injection_dates,
        injection_prices=injection_prices,
        withdrawal_dates=withdrawal_dates,
        withdrawal_prices=withdrawal_prices,
        pump_rate=pump_rate,
        max_volume=max_volume,
        storage_costs=storage_costs
    )

    print(f"\nContract Value: ${contract_value:,.2f}")
    print(f"ROI per MMBtu: ${contract_value / (pump_rate * len(injection_dates)):.2f}")
    
    return contract_value