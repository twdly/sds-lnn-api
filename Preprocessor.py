import numpy as np
import pandas as pd
import torch

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def preprocess_data(df):

    # Conversion values
    _63 = 63 / 1.852
    _116 = 116 / 1.852
    _88 = 88 / 1.852
    _117 = 117 / 1.852
    _159 = 159 / 1.852
    _199 = 199 / 1.852

    # Intensity thresholds
    intensity_thresholds = {
        0: (0, _63),
        1: (_63, _88),
        2: (_88, _117),
        3: (_117, _159),
        4: (_159, _199),
        5: (_199, 250)
    }

    def get_features(df):
        columns = ['ISO_TIME', 'BOM_LAT', 'BOM_LON', 'BOM_WIND', 'BOM_PRES', 'BOM_GUST', 'BOM_EYE', 'STORM_SPEED', 'NATURE']
        return df[columns].copy()

    def make_numeric(df):
        numeric_columns = ['BOM_LAT', 'BOM_LON', 'BOM_WIND', 'BOM_PRES', 'BOM_GUST', 'BOM_EYE', 'STORM_SPEED']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        return df.copy()

    def feature_select(df):
        features = get_features(df)
        features = make_numeric(features)
        features_cleaned = features.dropna(subset=['BOM_WIND'])
        return features_cleaned

    def classify_tropical_system(row):
        if row['NATURE'] == 'TS':
            wind_speed = row['BOM_WIND']
            if wind_speed < _63:
                return 'TD'
            elif _63 <= wind_speed < _116:
                return 'TS'
            elif wind_speed > _116:
                return 'TC'
        return row['NATURE']

    def assign_NR(row):
        windspeed = row['BOM_WIND']
        if row['NATURE'] == 'DS':
            return windspeed / _63
        if row['NATURE'] == 'TD':
            return windspeed / _63
        if row['NATURE'] == 'TS':
            return (windspeed - _63) / (_116 - _63)
        if row['NATURE'] == 'TC':
            return (windspeed - _116) / (250 - _116)
        else:
            return 0

    def assign_intensity(row):
        wind_speed = row['BOM_WIND']
        if _63 <= wind_speed <= _88:
            return 1
        elif _88 < wind_speed <= _117:
            return 2
        elif _117 < wind_speed <= _159:
            return 3
        elif _159 < wind_speed <= _199:
            return 4
        elif wind_speed > _199:
            return 5
        else:
            return 0

    def assign_IR(row):
        windspeed = row['BOM_WIND']
        for intensity, (low, high) in intensity_thresholds.items():
            if low <= windspeed < high:
                return (windspeed - low) / (high - low)
        return 0

    # Preprocessing steps
    df = feature_select(df)
    df = df[df['NATURE'].isin(['DS', 'TS'])]
    df['NATURE'] = df.apply(classify_tropical_system, axis=1)

    month_map = {
        1 : 'Jan',
        2 : 'Feb',
        3 : 'Mar',
        4 : 'Apr',
        5 : 'May',
        6 : 'Jun',
        7 : 'Jul',
        8 : 'Aug',
        9 : 'Sep',
        10 : 'Oct',
        11 : 'Nov',
        12 : 'Dec'
    }

    nature_mapping = {
        'DS': 0,
        'TD': 1,
        'TS': 2,
        'TC': 3
    }
    df['NATURE_ID'] = df['NATURE'].map(nature_mapping)
    df['NR'] = df.apply(assign_NR, axis=1)
    df['INTENSITY'] = df.apply(assign_intensity, axis=1)
    df['IR'] = df.apply(assign_IR, axis=1)
    df['N'] = df['NATURE_ID'] + df['NR']
    df['I'] = df['INTENSITY'] + df['IR']

    # Handling missing values
    df['BOM_EYE'] = df['BOM_EYE'].fillna(0)
    df['STORM_SPEED'] = df['STORM_SPEED'].fillna(0)
    df['BOM_GUST'] = df['BOM_GUST'].fillna(df['BOM_GUST'].mean())
    df['BOM_PRES'] = df['BOM_PRES'].fillna(df['BOM_PRES'].mean())
    df['BOM_LAT'] = df['BOM_LAT'].fillna('NaN')
    df['BOM_LON'] = df['BOM_LON'].fillna('NaN')
    df['BOM_LAT'] = pd.to_numeric(df['BOM_LAT'], errors='coerce')
    df['BOM_LON'] = pd.to_numeric(df['BOM_LON'], errors='coerce')

    # Extracting date features
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df['YEAR'] = df['ISO_TIME'].dt.year.astype('Int64').astype(str)
    df['MONTH'] = df['ISO_TIME'].dt.month.astype('Int64')
    df['MONTH_NAME'] = df['MONTH'].map(month_map)
    df['DAY'] = df['ISO_TIME'].dt.day.astype('Int64')
    df['HOUR'] = df['ISO_TIME'].dt.hour.astype('Int64')

    # Selecting final columns
    final_columns = ['ISO_TIME', 'YEAR', 'MONTH_NAME','MONTH','DAY', 'HOUR',  # time shit
                     'BOM_LAT', 'BOM_LON', 'BOM_WIND', 'BOM_PRES', 'BOM_GUST', 'BOM_EYE', 'STORM_SPEED',  # features
                     'NATURE', 'NATURE_ID', 'N', 'INTENSITY', 'I']  # classifications

    return df[final_columns]


def get_training_data(df, start_year, end_year):
    feature_columns = ['MONTH', 'DAY', 'HOUR', 'BOM_LAT', 'BOM_LON', 'BOM_WIND', 'BOM_PRES', 'BOM_GUST', 'BOM_EYE', 'STORM_SPEED']
    target_columns = ['N', 'I']
    # discrete_targets = ['NATURE_ID', 'INTENSITY']

    # Collect tensors for each feature, target, and discrete
    features_list = []
    target_list = []
    discrete_list = []

    for year in range(int(start_year), int(end_year) + 1):
        year_data = df[df['YEAR'] == str(year)]
        months = year_data['MONTH'].unique()

        # Split batches by consecutive months
        continuous_samples = []
        current_sample = [months[0]]
        for i in range(1, len(months)):
            if months[i] == months[i - 1] + 1:
                current_sample.append(months[i])
            else:
                continuous_samples.append(current_sample)
                current_sample = [months[i]]
        continuous_samples.append(current_sample)

        # For each sample, turn them into tensors
        for sample_months in continuous_samples:
            sample_data = year_data[year_data['MONTH'].isin(sample_months)]

            # Convert data to numpy and then to tensor explicitly
            try:
                # Features tensor (1 x T x F)
                I_numpy = sample_data[feature_columns].to_numpy(dtype=np.float32)  # Ensuring data is float32
                I_tensor = torch.tensor(I_numpy, dtype=torch.float32).unsqueeze(0).to(device)
                features_list.append(I_tensor)

                # Targets tensor (1 x T x O)
                target_numpy = sample_data[target_columns].to_numpy(dtype=np.float32)  # Ensuring data is float32
                target_tensor = torch.tensor(target_numpy, dtype=torch.float32).unsqueeze(0).to(device)
                target_list.append(target_tensor)

                # Discrete labels tensor (1 x T x O)
                # discrete_numpy = sample_data[discrete_targets].to_numpy(dtype=np.float32)  # Ensuring data is float32
                # discrete_tensor = torch.tensor(discrete_numpy, dtype=torch.float32).unsqueeze(0).to(device)
                # discrete_list.append(discrete_tensor)

            except Exception as e:
                print(f"Data type conversion error for year {year}: {e}")

    return features_list, target_list


def get_tensors_to_predict(df):  # only from 1973 to 2024
    # month day and hour should be int64, might want to change to float32
    # bom_lat
    feature_columns = ['MONTH', 'DAY', 'HOUR', 'BOM_LAT', 'BOM_LON', 'BOM_WIND', 'BOM_PRES', 'BOM_GUST', 'BOM_EYE', 'STORM_SPEED']
    # target_columns = ['N', 'I']

    # I_set = df[df['YEAR'].between(start_year, end_year)]
    I = torch.tensor(df[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    # target = torch.tensor(df[target_columns].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)

    # if you want to retrain model, then uncomment target and return target.unsqueeze(0)
    return I.unsqueeze(0)#, target.unsqueeze(0)


def get_tensors_to_predict_for_training(df, start_year, end_year):  # only from 1973 to 2024
    # month day and hour should be int64, might want to change to float32
    # bom_lat
    feature_columns = ['MONTH', 'DAY', 'HOUR', 'BOM_LAT', 'BOM_LON', 'BOM_WIND', 'BOM_PRES', 'BOM_GUST', 'BOM_EYE', 'STORM_SPEED']
    target_columns = ['N', 'I']

    I_set = df[df['YEAR'].between(start_year, end_year)]
    I = torch.tensor(I_set[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    target = torch.tensor(I_set[target_columns].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)

    # if you want to retrain model, then uncomment target and return target.unsqueeze(0)
    return I.unsqueeze(0), target.unsqueeze(0)
