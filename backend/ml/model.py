import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import datetime
import joblib
from core.logger import LoggerFactory
logger = LoggerFactory.get_logger()
import warnings


class PricePredictor:
    def __init__(self):
        logger.info("Initializing Enhanced PricePredictor...")
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.load_model()

    def load_model(self):
        try:
            model_data = joblib.load('models/price_predictor_model.joblib')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['features']
            self.metrics = model_data.get('metrics', None)  # Load metrics if they exist
            logger.info("Model and metrics loaded successfully.")
        except FileNotFoundError:
            logger.info("No existing model found. A new model will be trained.")
            self._initialize_model()
            train_data = pd.read_csv('ml/update_data.csv')
            self.train(train_data)

    def _initialize_model(self):
        # Using XGBoost with optimized parameters for speed and performance
        self.model = XGBRegressor(
            n_estimators=1000,  # Reduced from 1000
            learning_rate=0.03,  # Increased from 0.01
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

    def save_model(self):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.feature_columns,
            'metrics': self.metrics  # Save metrics along with the model
        }
        joblib.dump(model_data, 'models/price_predictor_model.joblib')
        logger.info("Model and metrics saved successfully.")

    def _filter_valid_data(self, df, is_training=True):
        """Filter out rows with invalid data formats"""
        original_len = len(df)
        
        logger.debug(f"Starting data filtering. Input shape: {df.shape}")
        logger.debug(f"Sample of time values:\n{df['time'].head() if 'time' in df.columns else 'No time column'}")
        
        # Check for missing columns - different requirements for training vs prediction
        required_cols = ['distance']
        if 'time' not in df.columns and 'hour' not in df.columns:
            required_cols.append('time')  # Need either time or hour
        if is_training:
            required_cols.append('new_price')
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Start with all rows valid
        valid_mask = pd.Series(True, index=df.index)
        
        # Validate time format if present
        if 'time' in df.columns:
            valid_time_mask = df['time'].astype(str).str.contains('^\d{1,2}:\d{2}(:\d{2})?$', regex=True)
            invalid_times = df[~valid_time_mask]['time'].unique()
            logger.debug(f"Invalid time formats found: {invalid_times[:10]}")
            valid_mask &= valid_time_mask
        
        # Validate numeric columns
        numeric_conditions = [
            df['distance'].notna(),
            (df['distance'] > 0)
        ]
        
        if is_training:
            numeric_conditions.extend([
                df['new_price'].notna(),
                (df['new_price'] > 0)
            ])
        
        valid_numeric_mask = pd.Series(True, index=df.index)
        for condition in numeric_conditions:
            valid_numeric_mask &= condition
        
        valid_mask &= valid_numeric_mask
        
        # Get filtered dataframe
        filtered_df = df[valid_mask].copy()
        
        # Log filtering results
        removed_count = original_len - len(filtered_df)
        if removed_count > 0:
            logger.warning(f"Filtered out {removed_count}/{original_len} rows with invalid data format")
            if 'time' in df.columns:
                logger.warning(f"Invalid time format: {(~valid_time_mask).sum()} rows")
            logger.warning(f"Invalid numeric data: {(~valid_numeric_mask).sum()} rows")
            
        logger.debug(f"Final filtered shape: {filtered_df.shape}")
        logger.debug(f"Sample of valid data:\n{filtered_df.head()}")
        
        return filtered_df

    def _engineer_features(self, df):
        """Engineer features from the input data"""
        # Filter invalid data first
        df = self._filter_valid_data(df, is_training='new_price' in df.columns)
        
        # Convert time to hour - now we know all times are valid
        if 'time' in df.columns:
            df['hour'] = df['time'].astype(str).str.split(':').str[0].astype(int)
        
        # Create time-based features
        if 'hour' in df.columns:
            df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
            df['is_peak_evening'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
            df['is_night'] = (((df['hour'] >= 22) & (df['hour'] <= 23)) | 
                             ((df['hour'] >= 0) & (df['hour'] <= 5))).astype(int)
        
        # Distance-based features
        df['distance_squared'] = df['distance'] ** 2
        df['log_distance'] = np.log1p(df['distance'])
        
        # Weather-based features (if available)
        if 'temperature' in df.columns:
            df['temp_extreme'] = ((df['temperature'] < 15) | 
                                (df['temperature'] > 35)).astype(int)
        
        if 'rain_amount' in df.columns:
            df['has_rain'] = (df['rain_amount'] > 0).astype(int)
            df['heavy_rain'] = (df['rain_amount'] > 25).astype(int)
        
        if 'wind_speed' in df.columns:
            df['strong_wind'] = (df['wind_speed'] > 10).astype(int)
        
        # Weather severity (if all weather columns are present)
        weather_cols = ['temperature', 'rain_amount', 'wind_speed']
        if all(col in df.columns for col in weather_cols):
            df['weather_severity'] = (
                df['temp_extreme'] + 
                df['has_rain'] * 1.5 + 
                df['heavy_rain'] * 2 + 
                df['strong_wind']
            )
        
        # Interaction features
        if 'weather_severity' in df.columns:
            df['distance_weather'] = df['distance'] * df['weather_severity']
        
        if 'hour' in df.columns:
            df['distance_peak'] = (
                df['distance'] * 
                (df['is_peak_morning'] + df['is_peak_evening'] * 1.2 + df['is_night'] * 0.8)
            )
        
        # Drop the original time column as we now have hour
        if 'time' in df.columns:
            df = df.drop('time', axis=1)
        
        return df

    def _convert_to_native_types(self, obj):
        """Convert numpy types to Python native types for serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            # Handle NaN values
            if np.isnan(obj):
                return 0.0  # or another default value
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_native_types(obj.tolist())
        return obj

    def _augment_hourly_data(self, df):
        """Augment data with additional samples for different hours"""
        df_augmented = df.copy()
        original_rows = df_augmented.to_dict('records')
        new_rows = []

        # Define price multipliers for different hours
        hour_multipliers = {
            # Morning peak (7-9): 20-40% increase
            7: 1.3,
            8: 1.4,
            9: 1.2,
            # Evening peak (16-18): 30-50% increase
            16: 1.3,
            17: 1.5,
            18: 1.3,
            # Late night (22-5): 10-20% increase
            22: 1.1,
            23: 1.15,
            0: 1.2,
            1: 1.2,
            2: 1.15,
            3: 1.1,
            4: 1.1,
            5: 1.1,
            # Normal hours: slight variations
            6: 1.0,
            10: 1.0,
            11: 0.95,
            12: 1.0,
            13: 1.0,
            14: 0.95,
            15: 1.1,
            19: 1.1,
            20: 1.05,
            21: 1.05
        }

        # Generate new samples for each hour
        for row in original_rows:
            base_price = row['new_price']
            base_time = row['time']
            
            # Create variations for each hour
            for hour in range(24):
                multiplier = hour_multipliers.get(hour, 1.0)
                
                # Add some random variation (±5%)
                random_factor = 1.0 + np.random.uniform(-0.05, 0.05)
                final_multiplier = multiplier * random_factor
                
                new_row = row.copy()
                new_row['time'] = f"{hour:02d}:00:00"
                new_row['new_price'] = base_price * final_multiplier
                
                # Add some randomness to weather conditions
                if hour >= 22 or hour <= 5:  # Night hours
                    new_row['temperature'] *= 0.9  # Slightly cooler at night
                elif 10 <= hour <= 16:  # Day hours
                    new_row['temperature'] *= 1.1  # Slightly warmer during day
                
                new_rows.append(new_row)

        # Add the new rows to the original dataframe
        augmented_df = pd.concat([df_augmented, pd.DataFrame(new_rows)], ignore_index=True)
        
        # Shuffle the data
        augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
        
        logger.info(f"Data augmented from {len(df)} to {len(augmented_df)} samples")
        
        # Log hour distribution
        hour_counts = augmented_df['time'].apply(lambda x: int(x.split(':')[0])).value_counts()
        logger.info(f"Hour distribution after augmentation:\n{hour_counts.sort_index()}")
        
        return augmented_df

    def _analyze_weather_impact(self, df, original_hours=None):
        """Analyze the impact of weather conditions on prices"""
        analysis_df = df.copy()
        
        if original_hours is not None:
            analysis_df['hour'] = original_hours
        
        # Calculate correlations with NaN handling
        def safe_correlation(x, y):
            try:
                corr = np.corrcoef(x, y)[0,1]
                return float(0.0) if np.isnan(corr) else float(corr)
            except:
                return float(0.0)
        
        weather_analysis = {
            'temp_correlation': safe_correlation(analysis_df['temperature'], analysis_df['new_price']),
            'rain_correlation': safe_correlation(analysis_df['rain_amount'], analysis_df['new_price']),
            'humidity_correlation': safe_correlation(analysis_df['humidity'], analysis_df['new_price']),
            'wind_correlation': safe_correlation(analysis_df['wind_speed'], analysis_df['new_price']),
        }
        
        # Convert pandas describe() output to native Python types with NaN handling
        def convert_describe(series_desc):
            stats_dict = series_desc.to_dict()
            return {k: float(0.0) if np.isnan(v) else float(v) 
                   for k, v in stats_dict.items()}
        
        # Handle empty groups in price distribution analysis
        def safe_describe(data):
            if len(data) == 0:
                return pd.Series({
                    'count': 0.0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    '25%': 0.0,
                    '50%': 0.0,
                    '75%': 0.0,
                    'max': 0.0
                })
            return data.describe()
        
        # Analyze price distribution by weather condition
        weather_analysis['price_by_weather'] = {
            'rainy': convert_describe(safe_describe(analysis_df[analysis_df['has_rain'] == 1]['new_price'])),
            'normal': convert_describe(safe_describe(analysis_df[analysis_df['has_rain'] == 0]['new_price'])),
            'extreme_temp': convert_describe(safe_describe(analysis_df[analysis_df['temp_extreme'] == 1]['new_price'])),
            'normal_temp': convert_describe(safe_describe(analysis_df[analysis_df['temp_extreme'] == 0]['new_price'])),
        }
        
        # Time-based analysis with improved hour handling
        logger.info("Analyzing price distribution by hour")
        hourly_prices = {}
        for hour in range(24):
            hour_data = analysis_df[analysis_df['hour'] == hour]['new_price']
            if len(hour_data) > 0:
                # Only include hours that have data
                mean_price = hour_data.mean()
                if not pd.isna(mean_price):
                    hourly_prices[str(hour)] = float(mean_price)
                else:
                    hourly_prices[str(hour)] = 0.0
            else:
                # Mark hours with no data differently
                hourly_prices[str(hour)] = None
        
        # Filter out None values and ensure we have at least some valid prices
        valid_prices = {k: v for k, v in hourly_prices.items() if v is not None}
        if not valid_prices:
            # If no valid prices, set all hours to 0
            weather_analysis['price_by_hour'] = {str(h): 0.0 for h in range(24)}
        else:
            # Only include hours with actual data
            weather_analysis['price_by_hour'] = valid_prices
        
        return weather_analysis

    def _analyze_hourly_prices(self, df, original_hours=None):
        """Analyze price variations by hour, normalized by distance"""
        logger.debug(f"Starting hourly price analysis. Input shape: {df.shape}")
        logger.debug(f"Columns available: {df.columns.tolist()}")
        
        # Get hours from the appropriate source
        if original_hours is not None:
            hours = original_hours
            logger.debug("Using provided original_hours")
        elif 'hour' in df.columns:
            hours = df['hour']
            logger.debug("Using existing hour column")
        elif 'time' in df.columns:
            logger.debug("Processing time column to extract hours")
            df = self._filter_valid_data(df, is_training=True)  # Always use training mode for analysis
            hours = df['time'].astype(str).str.split(':').str[0].astype(int)
        else:
            raise ValueError("No time or hour information available in the data")
            
        logger.debug(f"Unique hours found: {sorted(hours.unique())}")
        
        # Calculate price per kilometer for each entry
        price_per_km = df['new_price'] / df['distance']
        
        # Group by hour and calculate statistics
        hourly_stats = pd.DataFrame({
            'hour': hours,
            'price_per_km': price_per_km,
            'distance': df['distance'],
            'price': df['new_price']
        }).groupby('hour').agg({
            'price_per_km': ['mean', 'std', 'count'],
            'distance': ['mean', 'std'],
            'price': ['mean', 'std']
        })
        
        # Calculate confidence intervals
        confidence_level = 0.98
        z_score = 1.96  # for 95% confidence interval
        
        analysis = {}
        for hour in range(24):
            if hour in hourly_stats.index:
                stats = hourly_stats.loc[hour]
                n = stats[('price_per_km', 'count')]
                mean_price_per_km = stats[('price_per_km', 'mean')]
                std_price_per_km = stats[('price_per_km', 'std')]
                
                # Calculate confidence interval for price per km
                margin_of_error = z_score * (std_price_per_km / np.sqrt(n))
                ci_lower = mean_price_per_km - margin_of_error
                ci_upper = mean_price_per_km + margin_of_error
                
                # Calculate relative price factor compared to overall mean
                overall_mean_price_per_km = price_per_km.mean()
                price_factor = mean_price_per_km / overall_mean_price_per_km
                
                analysis[hour] = {
                    'mean_price_per_km': float(mean_price_per_km),
                    'std_price_per_km': float(std_price_per_km),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
                    'price_factor': float(price_factor),
                    'sample_count': int(n),
                    'mean_distance': float(stats[('distance', 'mean')]),
                    'mean_price': float(stats[('price', 'mean')]),
                    'is_peak': self._is_peak_hour(hour)
                }
            else:
                analysis[hour] = {
                    'mean_price_per_km': 0,
                    'std_price_per_km': 0,
                    'ci_lower': 0,
                    'ci_upper': 0,
                    'price_factor': 1.0,
                    'sample_count': 0,
                    'mean_distance': 0,
                    'mean_price': 0,
                    'is_peak': self._is_peak_hour(hour)
                }
        
        return analysis
    
    def _is_peak_hour(self, hour):
        """Determine if an hour is during peak time"""
        morning_peak = (7 <= hour <= 9)
        evening_peak = (16 <= hour <= 18)
        late_night = (22 <= hour <= 23) or (0 <= hour <= 5)
        return 'morning_peak' if morning_peak else 'evening_peak' if evening_peak else 'late_night' if late_night else 'normal'
    
    def _get_hourly_metrics(self):
        """Get comprehensive hourly price metrics"""
        if not hasattr(self, '_hourly_analysis'):
            return {}
            
        metrics = {
            'hourly_analysis': self._hourly_analysis,
            'peak_summary': {
                'morning_peak': {
                    'hours': '07:00-09:00',
                    'avg_price_factor': np.mean([
                        self._hourly_analysis[h]['price_factor'] 
                        for h in range(7, 10)
                    ])
                },
                'evening_peak': {
                    'hours': '16:00-18:00',
                    'avg_price_factor': np.mean([
                        self._hourly_analysis[h]['price_factor'] 
                        for h in range(16, 19)
                    ])
                },
                'late_night': {
                    'hours': '22:00-05:00',
                    'avg_price_factor': np.mean([
                        self._hourly_analysis[h]['price_factor'] 
                        for h in list(range(22, 24)) + list(range(0, 6))
                    ])
                }
            }
        }
        
        return metrics
    
    def train(self, X_train):
        """Train the model with the given data and compute comprehensive metrics"""
        logger.info("Starting model training with enhanced features...")
        
        # Engineer features
        logger.info("Engineering features...")
        df = self._engineer_features(X_train.copy())
        logger.info("Features engineered.")
        
        # Calculate hourly price analysis before scaling
        logger.info("Analyzing hourly price patterns...")
        self._hourly_analysis = self._analyze_hourly_prices(df)
        
        # Separate target
        y = df['new_price']
        X = df.drop('new_price', axis=1)
        
        # Store original hour data before scaling
        original_hours = X['hour'].copy()
        
        # Store feature columns for prediction
        self.feature_columns = X.columns
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data for validation
        logger.info("Splitting data for validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        

        logger.info("Training model...")
        # Train model
        self.model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        logger.info("Model training completed, calculating metrics...")
        # Calculate comprehensive metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        

        logger.info("Calculating feature importance...")
        # Normalize and filter feature importance
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        # Remove features with very low importance (less than 1% of max importance)
        max_importance = max(feature_importance.values())
        threshold = max_importance * 0.01
        feature_importance = {k: float(v) for k, v in feature_importance.items() 
                            if v > threshold}
        
        # Renormalize remaining features to sum to 1
        total_importance = sum(feature_importance.values())
        feature_importance = {k: float(v/total_importance) 
                            for k, v in feature_importance.items()}
        
        metrics = {
            'train_r2': float(r2_score(y_train, train_pred)),
            'val_r2': float(r2_score(y_val, val_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
            'feature_importance': feature_importance,
        }

        logger.info("Model training completed, metrics: %s", metrics)
        
        # Calculate weather impact analysis using original data
        weather_analysis = self._analyze_weather_impact(df, original_hours)
        metrics.update(weather_analysis)
        
        # Add hourly price analysis
        metrics.update(self._get_hourly_metrics())
        
        self.metrics = self._convert_to_native_types(metrics)
        self.save_model()  # Save both model and metrics
        return self.metrics

    def get_model_metrics(self):
        """Return comprehensive model metrics and analysis"""
        if not hasattr(self, 'metrics'):
            return None
        return self._convert_to_native_types(self.metrics)

    def predict(self, X_test):
        """Predict prices for new data"""
        # Engineer features
        df = self._engineer_features(X_test.copy())
        
        if 'new_price' in df.columns:
            df = df.drop(['new_price'], axis=1)
        
        # Ensure we have all required features in the correct order
        df = df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions

    def calculate_weather_correlations(self, data=None):
        """Calculate correlations between weather features and key metrics"""
        if data is None:
            data = pd.read_csv('ml/update_data.csv')
        
        # Engineer features before calculating correlations
        data = self._engineer_features(data)
        
        # Include distance and new_price in the analysis
        features = ['temperature', 'rain_amount', 'humidity', 'wind_speed', 'distance', 'new_price']
        analysis_data = data[features]
        
        correlations = {}
        corr_matrix = analysis_data.corr()
        
        # Focus on rain_amount, distance, and new_price correlations
        target_features = ['rain_amount', 'distance', 'new_price']
        
        for feat1 in target_features:
            for feat2 in features:
                if feat1 != feat2:
                    correlation = corr_matrix.loc[feat1, feat2]
                    spearman_corr = analysis_data[feat1].corr(analysis_data[feat2], method='spearman')
                    
                    correlations[f"{feat1}_vs_{feat2}"] = {
                        'pearson_correlation': correlation,
                        'spearman_correlation': spearman_corr,
                        'interpretation': self._interpret_correlation(correlation),
                        'strength': self._get_correlation_strength(correlation)
                    }
                    
                    # Add additional statistical insights for key relationships
                    if (feat1 == 'rain_amount' and feat2 in ['distance', 'new_price']) or \
                       (feat1 == 'distance' and feat2 == 'new_price'):
                        correlations[f"{feat1}_vs_{feat2}"].update({
                            'covariance': analysis_data[feat1].cov(analysis_data[feat2]),
                            'r_squared': correlation ** 2
                        })
        
        return correlations

    def _get_correlation_strength(self, correlation):
        """Determine the strength of correlation"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return 'Very Strong'
        elif abs_corr >= 0.6:
            return 'Strong'
        elif abs_corr >= 0.4:
            return 'Moderate'
        elif abs_corr >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'

    def _interpret_correlation(self, correlation):
        """Interpret the correlation coefficient"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"
            
        direction = "positive" if correlation >= 0 else "negative"
        return f"A {strength} {direction} correlation"

    def calculate_model_metrics(self, data=None):
        """Calculate various model performance metrics"""
        if data is None:
            data = pd.read_csv('ml/update_data.csv')
            
        # Engineer features before calculating metrics
        data = self._engineer_features(data)
        
        X = data[self.feature_columns]
        y = data['new_price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'interpretation': {
                'mae': f"On average, predictions are off by {mae:.2f} units",
                'rmse': f"Root mean squared error is {rmse:.2f} units",
                'mape': f"Average percentage error is {mape:.2f}%",
                'r2': f"Model explains {r2*100:.2f}% of price variance"
            }
        }
        
        return metrics
