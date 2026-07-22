import os
import shap
import boto3
import joblib
import random
import getpass
import sqlite3
import requests
import calendar
import pandas as pd
import xgboost as xgb
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from datetime import datetime, timedelta, timezone
from sklearn.pipeline import Pipeline
from EMReport import EMReport_Classification
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def get_weather_data(lat, lon, dt_unix):
    #Weather Data Collection
    launch_dt = datetime.fromtimestamp(dt_unix, tz=timezone.utc)
    date_str = launch_dt.strftime('%Y-%m-%d')
    hour_index = launch_dt.hour 
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=temperature_2m,wind_speed_10m,visibility"
    try:
        res = requests.get(url, timeout=10).json()
        return {
            "temp_c": res['hourly']['temperature_2m'][hour_index],
            "wind_speed_kmh": res['hourly']['wind_speed_10m'][hour_index],
            "visibility_m": res['hourly']['visibility'][hour_index]
        }
    except Exception:
        return {"temp_c": None, "wind_speed_kmh": None, "visibility_m": None}

def get_spacex_data():
    #SpaceX Data Collection
    print("Attempt to extract LIVE data from SpaceX API...")
    try:
        res_launch = requests.get("https://api.spacexdata.com/v4/launches/latest", timeout=10)
        res_launch.raise_for_status()
        l_data = res_launch.json()
        
        res_pad = requests.get(f"https://api.spacexdata.com/v4/launchpads/{l_data['launchpad']}", timeout=10)
        res_pad.raise_for_status()
        p_data = res_pad.json()
        
        payload_mass = None
        orbit = "Unknown"
        if l_data.get('payloads'):
            res_payload = requests.get(f"https://api.spacexdata.com/v4/payloads/{l_data['payloads'][0]}", timeout=10)
            if res_payload.status_code == 200:
                pay_data = res_payload.json()
                payload_mass = pay_data.get('mass_kg')
                orbit = pay_data.get('orbit')

        booster_version = "Unknown"
        landing_success = None
        landing_type = None
        
        if l_data.get('cores') and len(l_data['cores']) > 0:
            core_info = l_data['cores'][0]
            landing_success = core_info.get('landing_success')
            landing_type = core_info.get('landing_type')
            
            if core_info.get('core'):
                res_core = requests.get(f"https://api.spacexdata.com/v4/cores/{core_info['core']}", timeout=10)
                if res_core.status_code == 200:
                    booster_version = res_core.json().get('serial', 'Unknown')

        if landing_success is None and landing_type is None:
            landing_outcome = "None None"
        else:
            success_str = "True" if landing_success else "False"
            type_str = landing_type if landing_type else "None"
            landing_outcome = f"{success_str} {type_str}"
        
        lat = p_data.get('latitude')
        lon = p_data.get('longitude')
        dt_unix = l_data.get('date_unix')
        weather = get_weather_data(lat, lon, dt_unix)
    
        print("Successful Live Data Acquisition by SpaceX!")
        return [{
            "mission_name": l_data.get('name'),
            "flight_number": l_data.get('flight_number'),
            "launch_site": p_data.get('name', 'Unknown'),
            "lat": p_data.get('latitude'),
            "lon": p_data.get('longitude'),
            "dt_unix": l_data.get('date_unix'),
            "booster_version": booster_version,
            "payload_mass_kg": payload_mass,
            "orbit": orbit,
            "landing_outcome": landing_outcome,
            "temp_c": weather['temp_c'],
            "wind_speed_kmh": weather['wind_speed_kmh'],
            "visibility_m": weather['visibility_m']
        }]
        
    except Exception as e:
        print(f"Failed to acquire Live SpaceX data ({e}). Using Mock Data...")

        def generate_synthetic_spacex_data(num_records=200):
            #Synthetic Data Generation
            print(f"Generating and Saving {num_records} Synthetic Launches to the Database...")
    
            synthetic_data = []
            base_date = datetime(2018, 1, 1)
    
            for i in range(1, num_records + 1):
                base_date += timedelta(days=random.randint(5, 15))
                wind_speed_kmh = round(random.uniform(0.0, 30.0), 1)
                visibility_m = random.choice([10000, 10000, 8000, 5000, 2000])
                
                if wind_speed_kmh > 22.0 or visibility_m < 4000:
                    landing_outcome = f"False {random.choice(['ASDS', 'Ocean'])}"
                else:
                    landing_outcome = f"True {random.choice(['ASDS', 'RTLS'])}" if random.random() > 0.10 else "False ASDS"

                synthetic_data.append({
                    "mission_name": f"Synthetic-Mission-{i}", 
                    "flight_number": i + 500,
                    "launch_site": random.choice(["KSC LC 39A", "CCSFS SLC 40", "VAFB SLC 4E"]),
                    "lat": 28.5, 
                    "lon": -80.5, 
                    "dt_unix": int(base_date.timestamp()),
                    "booster_version": random.choice(["B1058", "B1060", "B1062", "B1067"]),
                    "payload_mass_kg": round(random.uniform(2000.0, 16000.0), 1),
                    "orbit": random.choice(["LEO", "ISS", "GTO", "VLEO"]), 
                    "landing_outcome": landing_outcome,
                    "temp_c": round(random.uniform(5.0, 35.0), 1), 
                    "wind_speed_kmh": wind_speed_kmh, 
                    "visibility_m": visibility_m
                })
        
            return synthetic_data
        return generate_synthetic_spacex_data(200)


def save_to_sqlite(record):
    #SQLite Database Insertion
    launch_dt = datetime.fromtimestamp(record['dt_unix'], tz=timezone.utc)
    launch_time_str = launch_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    conn = sqlite3.connect('master_spacex_weather.db', timeout=10)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ultimate_launches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            launch_time TEXT,
            mission_name TEXT,
            flight_number INTEGER,
            launch_site TEXT,
            booster_version TEXT,
            payload_mass_kg REAL,
            orbit TEXT,
            landing_outcome TEXT,
            temp_c REAL,
            wind_speed_kmh REAL,
            visibility_m INTEGER
        )
    ''')

    cursor.execute('''
        INSERT INTO ultimate_launches (
            launch_time, mission_name, flight_number, launch_site, 
            booster_version, payload_mass_kg, orbit, landing_outcome,
            temp_c, wind_speed_kmh, visibility_m
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        launch_time_str,
        record['mission_name'],
        record['flight_number'],
        record['launch_site'],
        record['booster_version'],
        record['payload_mass_kg'],
        record['orbit'],
        record['landing_outcome'],
        record['temp_c'],
        record['wind_speed_kmh'],
        record['visibility_m']
    ))

    conn.commit()
    conn.close()


def cleanup_database():
    #Database Cleanup
    conn = sqlite3.connect('master_spacex_weather.db')
    df = pd.read_sql_query("SELECT * FROM ultimate_launches", conn)
    missing_payload_values = df['payload_mass_kg'].isnull().sum()
    if missing_payload_values > 0:
        df['payload_mass_kg'] = df.groupby('booster_version')['payload_mass_kg'].transform(lambda x: x.fillna(x.mean()))
    df.to_sql('ultimate_launches', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return f"Database cleaned! Filled {missing_payload_values} missing payload_mass_kg values with booster version averages."


def statistical_Analysis():
    conn = sqlite3.connect('master_spacex_weather.db')
    df = pd.read_sql_query("SELECT * FROM ultimate_launches", conn)
    print("●●● STATISTICAL ANALYSIS ●●●")
    #Descriptive Statistical Analysis
    print(df.describe())
    #Landing Success/Failure Rate Analysis
    landing_success_count = df["landing_outcome"].str.contains("True").sum()
    landing_failure_count = df["landing_outcome"].str.contains("False").sum()
    print(f"Landing Success percentage: {(landing_success_count / len(df)) * 100:.2f}%")
    print(f"Landing Failure percentage: {(landing_failure_count / len(df)) * 100:.2f}%")
    #Maximum, Minimum, Average Value Analysis for Payload Mass
    payload_stat = df['payload_mass_kg'].min(), df['payload_mass_kg'].max(), df['payload_mass_kg'].mean()
    print(f"Payload Mass (kg) - Min: {payload_stat[0]}, Max: {payload_stat[1]}, Mean: {payload_stat[2]:.2f}")
    #Average Analysis for Temperature, Wind Speed ​​and Visibility
    weather_stat = df["temp_c"].mean(), df["wind_speed_kmh"].mean(), df["visibility_m"].mean()
    print(f"Weather Averages - Temp (C): {weather_stat[0]:.2f}, Wind Speed (km/h): {weather_stat[1]:.2f}, Visibility (m): {weather_stat[2]:.2f}")
    #Analysis of Launch Distribution by Launch Site
    launch_site_distribution = df['launch_site'].value_counts(normalize=True) * 100
    print(f"Launch Site Distribution (%): {launch_site_distribution.to_dict()}")
    
    #Inductive Statistical Analysis
    #Assumption of landing success based on wind speed.
    #H0: Wind speed does not affect landing success.
    #H1: Wind speed affects landing success.
    success_wind = df[df["landing_outcome"].str.contains("True")]["wind_speed_kmh"]
    failure_wind = df[df["landing_outcome"].str.contains("False")]["wind_speed_kmh"]
    t_stat, p_value = ttest_ind(success_wind, failure_wind, equal_var=False)
    if p_value < 0.05:
        print(f"Reject H0: Wind Speed significantly affects Landing Outcome (t-statistic = {t_stat:.4f}, p-value = {p_value:.4f})")
    else:
        print(f"Fail to Reject H0: Wind Speed does not significantly affect Landing Outcome (t-statistic = {t_stat:.4f}, p-value = {p_value:.4f})")
    
    #Landing success hypothesis based on launch site.
    #H0: Launch site does not affect landing success.
    #H1: Launch site does affect landing success.
    contingency_table = pd.crosstab(df["launch_site"], df["landing_outcome"].str.contains("True"))
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    if p_value < 0.05:
        print(f"Reject H0: Launch Site significantly affects Landing Outcome (chi2-statistic = {chi2_stat:.4f}, p-value = {p_value:.4f})")
    else:  
        print(f"Fail to Reject H0: Launch Site does not significantly affect Landing Outcome (chi2-statistic = {chi2_stat:.4f}, p-value = {p_value:.4f})")
    
    #Landing success hypothesis based on Payload Mass and Orbit. 
    #H0: Payload Mass does not affect landing success per orbit.
    #H1: Payload Mass affects landing success per orbit.
    anova_data = df[df["landing_outcome"].str.contains("True")].groupby("orbit")["payload_mass_kg"].apply(list)
    f_stat, p_value = stats.f_oneway(*anova_data)
    if p_value < 0.05:
        print(f"Reject H0: Payload Mass significantly affects Landing Outcome (F-statistic = {f_stat:.4f}, p-value = {p_value:.4f})")
    else:
        print(f"Fail to Reject H0: Payload Mass does not significantly affect Landing Outcome (F-statistic = {f_stat:.4f}, p-value = {p_value:.4f})")
    
    #Landing Failure Hypothesis based on Visibility.
    #H0: Visibility does not affect landing failure.
    #H1: Visibility affects landing failure.
    success_visibility = df[df["landing_outcome"].str.contains("True")]["visibility_m"]
    failure_visibility = df[df["landing_outcome"].str.contains("False")]["visibility_m"]
    t_stat, p_value = ttest_ind(success_visibility, failure_visibility, equal_var=False)
    if p_value < 0.05:
        print(f"Reject H0: Visibility significantly affects Landing Outcome (t-statistic = {t_stat:.4f}, p-value = {p_value:.4f})")
    else:
        print(f"Fail to Reject H0: Visibility does not significantly affect Landing Outcome (t-statistic = {t_stat:.4f}, p-value = {p_value:.4f})")
    #Does the landing failure rate increase when visibility drops (e.g., < 5000 m)?
    df["low_visibility"] = df["visibility_m"] < 5000
    contingency_table = pd.crosstab(df["low_visibility"], df["landing_outcome"].str.contains("True"))
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    if p_value < 0.05:
        print(f"Reject H0: Low Visibility (<5000 m) significantly affects Landing Outcome (chi2-statistic = {chi2_stat:.4f}, p-value = {p_value:.4f})")
    else:
        print(f"Fail to Reject H0: Low Visibility (<5000 m) does not significantly affect Landing Outcome (chi2-statistic = {chi2_stat:.4f}, p-value = {p_value:.4f})")
    
    #Landing Success Hypothesis based on Payload Mass.
    #H0: Payload Mass does not affect landing success.
    #H1: Payload Mass affects landing success.
    success_payload = df[df["landing_outcome"].str.contains("True")]["payload_mass_kg"]
    failure_payload = df[df["landing_outcome"].str.contains("False")]["payload_mass_kg"]
    t_stat, p_value = ttest_ind(success_payload, failure_payload, equal_var=False)
    if p_value < 0.05:
        print(f"Reject H0: Payload Mass significantly affects Landing Outcome (t-statistic = {t_stat:.4f}, p-value = {p_value:.4f})")
    else:
        print(f"Fail to Reject H0: Payload Mass does not significantly affect Landing Outcome (t-statistic = {t_stat:.4f}, p-value = {p_value:.4f})")

    #Time Series Analysis to see if there is a trend over time.   
    df['success'] = df['landing_outcome'].str.contains("True").astype(int)
    df['cumulative_success_rate'] = df['success'].expanding().mean()
    corr, p_value = pearsonr(df['flight_number'], df['cumulative_success_rate'])
    if corr > 0.5 and p_value < 0.05:
        print(f"SpaceX if learning from its mistakes. The Success rate Increases, (Correlation: {corr:.2f}), (P-Value: {p_value:.4f}) ")
    else:
        print(f"No strong upward trend is observed over time. The Success rate doesn't Increases, (Correlation: {corr:.2f}, (P-Value: {p_value:.4f}) )")

    #Find Load Weight Per Year.
    df['launch_time'] = pd.to_datetime(df['launch_time'])
    df['cumulative_payload_mass'] = df['payload_mass_kg'].expanding().mean()
    corr, p_value = pearsonr(df['flight_number'], df['cumulative_payload_mass'])
    if corr > 0.5 and p_value < 0.05:
        print(f"SpaceX payload mass increases with the passage of time, (Correlation: {corr:.2f}), (P-Value: {p_value:.4f})")
    else:
        print(f"SpaceX payload mass didn't increase with the passage of time, (Correlation: {corr:.2f}), (P-Value: {p_value:.4f})")

    #Seasonality Assumption in Launches.
    #H0: Launches are Independent of Seasonality.
    #H1: Launches are not Independent of Seasonality.
    df['month'] = df['launch_time'].dt.month
    df['month_name'] = df['month'].apply(lambda x: calendar.month_abbr[x])
    month_counts = df['month'].value_counts().sort_index()
    expected_counts = [len(df) / 12] * 12
    chi2_stat, p_value = stats.chisquare(f_obs=month_counts, f_exp=expected_counts)
    if p_value < 0.05:
        print(f"Reject H0: Launches are not independent of seasonality, (P-Value: {p_value:.4f})")
    else:
        print(f"Fail to Reject H0: Launches are independent of seasonality, (P-Value: {p_value:.4f})")
    conn.commit()
    conn.close()

def encoding_modeling():
    conn = sqlite3.connect('master_spacex_weather.db')
    df = pd.read_sql_query("SELECT * FROM ultimate_launches", conn)
    le = LabelEncoder()
    ss = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore')
    preprocess = ColumnTransformer(transformers=[
        ('cat', ohe, ['launch_site', 'booster_version', 'orbit']),
        ('num', ss, ['payload_mass_kg', 'temp_c', 'wind_speed_kmh', 'visibility_m'])
        ])
    x = df[["launch_site", "booster_version", "orbit", "payload_mass_kg", "temp_c", "wind_speed_kmh", "visibility_m"]]
    y = df["landing_outcome"]
    y_encoded = le.fit_transform(y)
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    model = Pipeline(steps=[("preprocess", preprocess),("classifier", xgb.XGBClassifier(random_state=42))])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(EMReport_Classification(y_test, y_pred))
    param_grid = {
    'classifier__max_depth': [2, 3, 4],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__n_estimators': [50, 100, 150],
    'classifier__subsample': [0.7, 0.8, 1.0]       
    }
    inner_cv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=inner_cv,        
    verbose=1,
    n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    train_accuracy = best_model.score(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test)
    print(f"New Train Accuracy: {train_accuracy:.2f}")
    print(f"New Test Accuracy: {test_accuracy:.2f}")
    print(EMReport_Classification(y_test, best_model.predict(x_test)))
    xgb_model = best_model.named_steps['classifier']
    preprocessor = best_model.named_steps['preprocess']
    x_test_processed = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    x_test_shap = pd.DataFrame(x_test_processed, columns=feature_names)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(x_test_shap)
    joblib.dump(best_model, "best_model.pkl")
    print("The model has been saved to the file 'best_model.pkl'")
    print("\n" + "="*50)
    print(" Connecting with AWS S3 Bucket...")
    print("="*50)
    print("Please enter your AWS login details.\n")
    print("(If you simply press Enter, an attempt will be made to use Environment Variables)\n")
    aws_access_key = input("AWS Access Key ID: ").strip() or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = getpass.getpass("AWS Secret Access Key (Secret): ").strip() or os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = input("S3 Bucket name (e.g. spacex-ml-models): ").strip()
    region_name = input("Region (for example eu-central-1) [Default: eu-central-1]: ").strip() or "eu-central-1"
    if aws_access_key and aws_secret_key and bucket_name:
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name
            )
            
            s3_file_path = "production/best_model.pkl"
            print(f"\n Uploading model to bucket '{bucket_name}'...")
            s3_client.upload_file("best_model.pkl", bucket_name, s3_file_path)
            print(f"SUCCESS! The model has been uploaded to S3 and is ready for the Web App.")
        except Exception as e:
            print(f"\n Failed to upload the model to S3 ({e}). The model will be saved locally.")
    else:
        print("\n Upload to S3 was omitted due to missing data (the model remained only locally).")

    conn.commit()
    conn.close()




if __name__ == "__main__":
    #Data Pipeline
    print(" ●●● START SPACEX DATA PIPELINE ●●● \n")

    space_records = get_spacex_data()
    print(f"Entries {len(space_records)} have been received. Registering in the database....")
    for record in space_records:  
        save_to_sqlite(record)
    
    print("\n ●●● PREPARING DATA FOR MACHINE LEARNING ●●●")
    cleanup_database()

    statistical_Analysis()

    encoding_modeling()
    
    print("\n ●●● THE PIPELINE IS COMPLETED! ●●●")  
