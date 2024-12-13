import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_price(distance, hour, temperature, rain_amount, wind_speed):
    """Generate realistic price based on Grab-like pricing model"""
    # Base price calculation
    if distance <= 2:
        base_price = 10000 + (distance * 12000)
    elif distance <= 5:
        base_price = 10000 + (2 * 12000) + ((distance - 2) * 13200)  # 10% increase
    else:
        base_price = 10000 + (2 * 12000) + (3 * 13200) + ((distance - 5) * 14400)  # 20% increase
    
    # Time multiplier
    time_multiplier = 1.0
    if 7 <= hour <= 9:  # Morning rush
        time_multiplier = 1.4
    elif 17 <= hour <= 19:  # Evening rush
        time_multiplier = 1.45

    
    # Weather multiplier
    weather_multiplier = 1.0
    if rain_amount > 0:
        # Exponential increase based on rain amount
        weather_multiplier += min(0.6, rain_amount * 0.3)  # Up to 60% increase for heavy rain
        if rain_amount > 3:  # Heavy rain
            weather_multiplier += 0.2  # Additional surge for very heavy rain
    
    # Temperature impact
    if temperature > 35:  # Hot weather
        weather_multiplier += 0.15
    elif temperature < 15:  # Cold weather
        weather_multiplier += 0.1
    
    # Wind impact (5% for strong winds)
    if wind_speed > 20:
        weather_multiplier *= 1.05
    
    # Cap weather multiplier
    weather_multiplier = min(weather_multiplier, 1.3)
    
    # Calculate final price
    price = base_price * time_multiplier * weather_multiplier
    
    # Add small random variation (±2%)
    variation = np.random.uniform(0.98, 1.02)
    price *= variation
    
    return int(round(price, -3))  # Round to nearest 1000 VND

def generate_weather_data(csv_path, target_rows=100000):
    """Generate weather data and combine with existing taxi data"""
    # Read the base data
    df = pd.read_csv(csv_path)
    
    # Convert price and distance columns to numeric
    df['new_price'] = df['new_price'].astype(float)
    df['distance'] = df['distance'].astype(float)
    
    # Convert time to datetime for easier manipulation
    # Add seconds to time if not present
    df['time'] = df['time'].apply(lambda x: x + ":00" if len(x.split(':')) == 2 else x)
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
    
    # Calculate how many copies we need
    num_copies = (target_rows + len(df) - 1) // len(df)
    
    # Create copies of the dataframe
    dfs = [df.copy() for _ in range(num_copies)]
    
    # For each copy after the first, add some random variation
    for df_copy in dfs[1:]:
        # Add random variation to distance (±10%)
        df_copy['distance'] *= np.random.uniform(0.9, 1.1, size=len(df_copy))
        
        # Add random variation to price (±15%)
        df_copy['new_price'] *= np.random.uniform(0.85, 1.15, size=len(df_copy))
    
    # Combine all copies
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Trim to target size
    df_combined = df_combined.iloc[:target_rows]
    
    # Generate weather data
    num_rows = len(df_combined)
    
    # Extract hour from time string for temperature generation
    df_combined['hour'] = df_combined['time'].apply(lambda x: int(x.split(':')[0]))
    
    # Temperature (°C): More realistic daily pattern
    base_temp = np.random.normal(28, 3, num_rows)  # Base temperature around 28°C
    
    # Hour-based temperature adjustments
    hour_temp_adj = {
        'dawn': (4, 6, -4),      # Coolest before sunrise
        'morning': (7, 11, -2),   # Cool in morning
        'noon': (12, 14, 4),     # Hot at noon
        'afternoon': (15, 17, 5), # Hottest in afternoon
        'evening': (18, 20, 2),   # Cooling down
        'night': (21, 3, -3),    # Cool at night
    }
    
    # Apply hour-based temperature adjustments
    for period, (start, end, adj) in hour_temp_adj.items():
        if start <= end:
            mask = (df_combined['hour'] >= start) & (df_combined['hour'] <= end)
        else:  # Handle overnight periods
            mask = (df_combined['hour'] >= start) | (df_combined['hour'] <= end)
        base_temp[mask] += adj
    
    df_combined['temperature'] = base_temp.clip(15, 40)  # Limit to reasonable range
    
    # Rain amount (mm): More realistic distribution
    rain_prob = np.random.random(num_rows)
    df_combined['rain_amount'] = np.where(
        rain_prob < 0.7,  # 70% chance of no rain
        0,
        np.random.exponential(10, num_rows)  # Heavier rain when it occurs
    )
    df_combined['rain_amount'] = df_combined['rain_amount'].clip(0, 50)
    
    # Wind speed (m/s): More variation
    df_combined['wind_speed'] = np.random.gamma(2, 2, num_rows)
    df_combined['wind_speed'] = df_combined['wind_speed'].clip(0, 20)
    
    # Humidity (%): Correlated with temperature and rain
    base_humidity = 100 - (df_combined['temperature'] - 15) * 2  # Base humidity inversely related to temperature
    rain_effect = df_combined['rain_amount'] * 2  # Rain increases humidity
    df_combined['humidity'] = (base_humidity + rain_effect + np.random.normal(0, 5, num_rows)).clip(40, 100)
    
    # Store original prices for comparison
    original_prices = df_combined['new_price'].copy()
    
    # Calculate weather severity score (0-1 scale)
    weather_severity = (
        ((df_combined['temperature'] - 25).abs() / 15) * 0.3 +  # Temperature deviation from comfortable 25°C
        (df_combined['rain_amount'] / 50) * 0.4 +              # Rain amount (normalized)
        (df_combined['wind_speed'] / 20) * 0.3                 # Wind speed (normalized)
    )
    
    # Ensure harsh conditions always have higher prices
    # Split data into normal and harsh conditions
    harsh_mask = weather_severity > 0.6  # Define harsh conditions
    normal_mask = ~harsh_mask
    
    # Calculate price adjustments
    normal_prices = original_prices[normal_mask]
    harsh_adjustment = np.maximum(
        1.3,  # Minimum 30% increase
        1.3 + (weather_severity[harsh_mask] - 0.6) * 1.0  # Additional increase based on severity
    )
    
    # Apply price adjustments
    df_combined.loc[harsh_mask, 'new_price'] = original_prices[harsh_mask] * harsh_adjustment
    
    # Ensure harsh condition prices are always higher than normal condition maximums
    normal_max_price = normal_prices.max()
    harsh_prices = df_combined.loc[harsh_mask, 'new_price']
    if len(harsh_prices) > 0:
        min_harsh_price = harsh_prices.min()
        if min_harsh_price <= normal_max_price:
            # Adjust harsh prices to ensure they're higher
            price_increase = (normal_max_price - min_harsh_price) + 10000  # Add buffer
            df_combined.loc[harsh_mask, 'new_price'] += price_increase
    
    # Round numerical columns to reasonable precision
    df_combined['temperature'] = df_combined['temperature'].round(1)
    df_combined['rain_amount'] = df_combined['rain_amount'].round(1)
    df_combined['wind_speed'] = df_combined['wind_speed'].round(1)
    df_combined['humidity'] = df_combined['humidity'].round(1)
    df_combined['new_price'] = df_combined['new_price'].round(2)
    df_combined['distance'] = df_combined['distance'].round(2)
    
    # Drop the temporary hour column
    df_combined = df_combined.drop('hour', axis=1)
    
    return df_combined

def generate_sample(num_samples=10000):
    """Generate sample taxi ride data"""
    np.random.seed(42)  # For reproducibility
    
    # Generate random times throughout the day
    hours = np.random.randint(0, 24, num_samples)
    minutes = np.random.randint(0, 60, num_samples)
    times = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]
    
    # Generate distances (km) - log-normal distribution
    distances = np.random.lognormal(1.5, 0.5, num_samples)
    distances = np.clip(distances, 1, 30)  # Clip to reasonable range
    
    # Base price calculation
    base_rate = 15000  # Base rate per km
    base_prices = distances * base_rate
    
    # Add time-based price variations
    time_multipliers = []
    for h in hours:
        if 7 <= h <= 9:  # Morning peak
            mult = np.random.uniform(1.3, 1.5)
        elif 16 <= h <= 18:  # Evening peak
            mult = np.random.uniform(1.4, 1.6)
        elif 22 <= h or h <= 5:  # Late night
            mult = np.random.uniform(1.1, 1.3)
        else:  # Normal hours
            mult = np.random.uniform(0.9, 1.1)
        time_multipliers.append(mult)
    
    # Apply time multipliers
    final_prices = base_prices * time_multipliers
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        'distance': distances.round(2),
        'new_price': final_prices.round(2)
    })
    
    return df

if __name__ == "__main__":
    # Generate and save training data
    df = generate_sample(num_samples=10000)
    df.to_csv("data_sample.csv", index=False)
    print("Generated training data saved to data_sample.csv")
    
    df = generate_weather_data('data_sample.csv', target_rows=100000)
    # Save to new CSV file
    df.to_csv('update_data.csv', index=False)
    print(f"Generated {len(df)} rows of data and saved to update_data.csv")