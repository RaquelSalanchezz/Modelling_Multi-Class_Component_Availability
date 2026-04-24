"""
clustering.py — Data generation + clustering pipeline for GSMCA.
Called as an external Python 3 process from Java.

Usage:
    python clustering.py <output_kmeans_csv> <output_dbscan_csv> <scenario>

Arguments:
    output_kmeans_csv   Path where the KMeans clustered CSV will be saved
    output_dbscan_csv   Path where the DBSCAN clustered CSV will be saved
    scenario            Scenario to run: 'vehicles' or 'patients'

Example:
    python clustering.py vehicles_kmeans.csv vehicles_dbscan.csv vehicles
    python clustering.py companions_kmeans.csv companions_dbscan.csv patients
"""

import sys
import random
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==============================================================================
# SCENARIO: VEHICLES
# ==============================================================================

def generate_final_dataset(total_samples=20, noise_ratio=0.15, noise_mode="hard"):
    data = []
    for i in range(total_samples):
        is_punctual = i < (total_samples // 2)
        is_noisy = random.random() < noise_ratio

        # Base profiles
        if is_punctual:
            prof, o_dist, t_start, t_end, c_age, c_cat, c_cond = ("worker", 5, 7, 16, 2, "premium", "excellent")
        else:
            prof, o_dist, t_start, t_end, c_age, c_cat, c_cond = (random.choice(["student", "retired"]), 30, 11, 21, 12, "low_end", "poor")

        # Noise injection
        if is_noisy and noise_mode == "hard":
            o_dist  = random.uniform(80, 150)
            t_start = random.uniform(0, 24)
            prof    = "student" if is_punctual else "worker"
            c_cat   = "low_end" if is_punctual else "premium"

        t_start_val = random.gauss(t_start, 0.5)
        t_end_val   = random.gauss(t_end, 0.5)

        # Clamp to [0, 24] and ensure minimum window of 1 hour
        t_start_val = max(0.0, min(23.0, t_start_val))
        t_end_val   = max(t_start_val + 1.0, min(24.0, t_end_val))
        window      = t_end_val - t_start_val

        # Vehicle physical columns
        battery_capacity = random.choice([80, 100, 120, 150])
        charge_speed     = random.randint(5, 14)
        discharge_rate   = random.randint(3, 9)
        current_charge   = int(random.uniform(0.01, 0.25) * battery_capacity)
        distance         = random.randint(50, 300)
        driver_age       = random.randint(18, 70)
        use_frequency    = random.randint(1, 7)
        origin_dist_val  = max(0, random.gauss(o_dist, 1.5))

        # Derived fields
        battery_level   = current_charge / battery_capacity
        required_energy = max(0, battery_capacity - current_charge)
        required_time   = required_energy / charge_speed if required_energy > 0 else 0.0
        stress_index    = required_time / window if window > 0 else 0.0
        window_length   = max(1, int(round(window)))

        hour = int(t_start_val)
        if 6 <= hour < 12:
            tod = "morning"
        elif 12 <= hour < 17:
            tod = "afternoon"
        elif 17 <= hour < 21:
            tod = "evening"
        else:
            tod = "night"

        data.append({
            "distance":             distance,
            "available_time_start": int(round(t_start_val)),
            "available_time_end":   int(round(t_end_val)),
            "current_charge":       current_charge,
            "battery_capacity":     battery_capacity,
            "charge_speed":         charge_speed,
            "discharge_rate":       discharge_rate,
            "driver_age":           driver_age,
            "driver_profession":    prof,
            "use_frequency":        use_frequency,
            "origin_distance":      int(round(origin_dist_val)),
            "battery_level":        battery_level,
            "window_length":        window_length,
            "required_energy":      required_energy,
            "required_time":        required_time,
            "stress_index":         stress_index,
            "time_of_day":          tod,
            # Internal clustering columns
            "_origin_distance_raw":      origin_dist_val,
            "_available_time_start_raw": t_start_val,
            "_available_time_end_raw":   t_end_val,
            "_car_age":                  max(0, random.gauss(c_age, 1.2)),
            "_car_category":             c_cat,
            "_car_condition":            c_cond,
            "driver_profession_clust":   prof,
            "punctuality_group":         "punctual" if is_punctual else "late",
        })
    return pd.DataFrame(data)


def run_clustering_vehicles(df):
    X = pd.DataFrame({
        "origin_distance":      df["_origin_distance_raw"],
        "driver_profession":    df["driver_profession_clust"],
        "available_time_start": df["_available_time_start_raw"],
        "available_time_end":   df["_available_time_end_raw"],
        "car_age":              df["_car_age"],
        "car_category":         df["_car_category"],
        "car_condition":        df["_car_condition"],
    })

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['origin_distance', 'available_time_start', 'available_time_end', 'car_age']),
        ('cat', OneHotEncoder(),  ['driver_profession', 'car_category', 'car_condition'])
    ])
    X_processed = preprocessor.fit_transform(X)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    df['cluster_km'] = kmeans.fit_predict(X_processed)

    dbscan = DBSCAN(eps=1.8, min_samples=5)
    df['cluster_db'] = dbscan.fit_predict(X_processed)

    return df


OUTPUT_COLS_VEHICLES = [
    'distance', 'available_time_start', 'available_time_end',
    'current_charge', 'battery_capacity', 'charge_speed', 'discharge_rate',
    'driver_age', 'driver_profession', 'use_frequency', 'origin_distance',
    'battery_level', 'window_length', 'required_energy', 'required_time',
    'stress_index', 'time_of_day'
]


# ==============================================================================
# SCENARIO: PATIENTS (companions)
# ==============================================================================

def generate_companion_dataset(total_samples=400, noise_ratio=0.15, noise_mode="hard"):
    data = []
    for i in range(total_samples):
        is_punctual = i < (total_samples // 2)
        is_noisy = random.random() < noise_ratio

        # Base profiles
        # Punctual: hospital_staff/professional_carer, direct closeness,
        #           high visit frequency, short distance, few past delays
        # Late: family, extended closeness,
        #       low visit frequency, long distance, more past delays
        if is_punctual:
            c_type, closeness, o_dist, freq_mu, delays_mu = (
                random.choice(["hospital_staff", "professional_carer"]),
                "direct", 5, 6, 0.5
            )
        else:
            c_type, closeness, o_dist, freq_mu, delays_mu = (
                "family", "extended", 30, 2, 3.5
            )

        # Noise injection
        if is_noisy and noise_mode == "hard":
            o_dist    = random.uniform(40, 80)
            c_type    = "family" if is_punctual else random.choice(["hospital_staff", "professional_carer"])
            closeness = "extended" if is_punctual else "direct"
            freq_mu   = random.uniform(1, 7)
            delays_mu = random.uniform(0, 5)

        # family_closeness and visit_frequency are correlated
        if closeness == "direct":
            freq_mu = min(7, freq_mu + 1.0)
            o_dist  = max(1, o_dist - 3)
        else:
            freq_mu = max(1, freq_mu - 1.0)
            o_dist  = o_dist + 3

        # Original dataset variables
        room_distance      = random.randint(100, 220)
        start_hour         = random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16])
        end_hour           = start_hour + random.randint(1, 3)
        end_hour           = min(end_hour, 18)
        has_meal           = random.randint(0, 1)

        # Clustering variables
        origin_dist_val     = max(0, random.gauss(o_dist, 1.5))
        visit_frequency     = max(1, min(7, int(round(random.gauss(freq_mu, 1)))))
        punctuality_history = round(max(0.0, random.gauss(delays_mu, 0.8)), 2)

        data.append({
            # Original dataset columns
            "distance_room":        room_distance,
            "available_time_start": start_hour,
            "available_time_end":   end_hour,
            "has-food":             has_meal,
            # Punctuality clustering columns
            "companion_type":       c_type,
            "family_closeness":     closeness,
            "visit_frequency":      visit_frequency,
            "origin_distance":      int(round(origin_dist_val)),
            "punctuality_history":  punctuality_history,
            # Internal clustering columns (raw values for preprocessing)
            "_origin_distance_raw":     origin_dist_val,
            "_visit_frequency_raw":     visit_frequency,
            "_punctuality_history_raw": punctuality_history,
            "_companion_type":          c_type,
            "_family_closeness":        closeness,
            "punctuality_group":        "punctual" if is_punctual else "late",
        })
    return pd.DataFrame(data)


def run_clustering_patients(df):
    X = pd.DataFrame({
        "origin_distance":      df["_origin_distance_raw"],
        "visit_frequency":      df["_visit_frequency_raw"],
        "punctuality_history":  df["_punctuality_history_raw"],
        "companion_type":       df["_companion_type"],
        "family_closeness":     df["_family_closeness"],
    })

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['origin_distance', 'visit_frequency', 'punctuality_history']),
        ('cat', OneHotEncoder(), ['companion_type', 'family_closeness'])
    ])
    X_processed = preprocessor.fit_transform(X)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    df['cluster_km'] = kmeans.fit_predict(X_processed)

    dbscan = DBSCAN(eps=1.8, min_samples=5)
    df['cluster_db'] = dbscan.fit_predict(X_processed)

    return df


OUTPUT_COLS_PATIENTS = [
    'distance_room', 'available_time_start', 'available_time_end', 'has-food',
    'companion_type', 'family_closeness', 'visit_frequency',
    'origin_distance', 'punctuality_history'
]


# ==============================================================================
# SHARED: LABELLING AND SIGMA ASSIGNMENT
# (identical logic for both scenarios)
# ==============================================================================

def apply_labels_and_sigma(dataframe, cluster_col, is_dbscan=False):
    # Identify the punctual cluster by lowest mean origin_distance
    # Exclude noise points (-1) from the mean calculation
    valid_data = dataframe[dataframe[cluster_col] != -1]
    cluster_punctual = valid_data.groupby(cluster_col)['_origin_distance_raw'].mean().idxmin()

    def get_sigma(val):
        if val == cluster_punctual:
            return 0.25
        if is_dbscan and val == -1:
            return 1.0           # Noise: maximum uncertainty
        return 0.5

    def get_cluster_id(val):
        if val == cluster_punctual:
            return 0
        if is_dbscan and val == -1:
            return 2
        return 1

    def get_label(val):
        if val == cluster_punctual:
            return 'punctual'
        if is_dbscan and val == -1:
            return 'outlier/noise'
        return 'late'

    suffix = 'dbscan' if is_dbscan else 'kmeans'
    dataframe[f'label_{suffix}']   = dataframe[cluster_col].apply(get_label)
    dataframe[f'std_{suffix}']     = dataframe[cluster_col].apply(get_sigma)
    dataframe[f'cluster_{suffix}'] = dataframe[cluster_col].apply(get_cluster_id)


def save_output(dataframe, output_cols, cluster_col, std_col, filename):
    out = dataframe[output_cols].copy().reset_index(drop=True)
    out.insert(0, 'id', range(1, len(out) + 1))
    out['cluster']      = dataframe[cluster_col].values
    out['assigned_std'] = dataframe[std_col].values
    out.to_csv(filename, index=False)
    print(f"File '{filename}' created with {len(out)} rows.")
    print(f"  cluster distribution: {out['cluster'].value_counts().to_dict()}")
    print(f"  std distribution:     {out['assigned_std'].value_counts().to_dict()}\n")


# ==============================================================================
# MAIN — called from Java as external process
# ==============================================================================
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python clustering.py <output_kmeans_csv> <output_dbscan_csv> <scenario>")
        print("  scenario: 'vehicles' or 'patients'")
        sys.exit(1)

    output_kmeans = sys.argv[1]
    output_dbscan = sys.argv[2]
    scenario      = sys.argv[3].lower()

    if scenario == 'vehicles':
        print("Scenario: VEHICLES")
        df = generate_final_dataset()
        df = run_clustering_vehicles(df)
        apply_labels_and_sigma(df, 'cluster_km', is_dbscan=False)
        apply_labels_and_sigma(df, 'cluster_db', is_dbscan=True)
        save_output(df, OUTPUT_COLS_VEHICLES, 'cluster_kmeans', 'std_kmeans', output_kmeans)
        save_output(df, OUTPUT_COLS_VEHICLES, 'cluster_dbscan', 'std_dbscan', output_dbscan)

    elif scenario == 'patients':
        print("Scenario: PATIENTS")
        df = generate_companion_dataset()
        df = run_clustering_patients(df)
        apply_labels_and_sigma(df, 'cluster_km', is_dbscan=False)
        apply_labels_and_sigma(df, 'cluster_db', is_dbscan=True)
        save_output(df, OUTPUT_COLS_PATIENTS, 'cluster_kmeans', 'std_kmeans', output_kmeans)
        save_output(df, OUTPUT_COLS_PATIENTS, 'cluster_dbscan', 'std_dbscan', output_dbscan)

    else:
        print(f"ERROR: Unknown scenario '{scenario}'. Use 'vehicles' or 'patients'.")
        sys.exit(1)

    print("Done. Both files generated.")
    print("\nExample noise assignments (DBSCAN):")
    print(df[df['cluster_db'] == -1][['punctuality_group', 'label_dbscan', 'std_dbscan']].head())
