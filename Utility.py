def min_max_feature_scaling(data):
    """Re-scale the range of features to scale the range in [0, 1]
    """
    # Min and max values for columns
    hr_mean_min_pre_norm = 50.777210
    hr_mean_max_pre_norm = 140.080622

    hr_std_min_pre_norm = 3.636572
    hr_std_max_pre_norm = 27.947662   

    # Min-max feature scaling
    # z = (x - min)/(max - min)
    data_min_max_scaled = data.drop(["index","subject","label"],axis=1)
    for column in data_min_max_scaled.columns:
        data_min_max_scaled[column] = (data_min_max_scaled[column] - data_min_max_scaled[column].min()) / (data_min_max_scaled[column].max() - data_min_max_scaled[column].min())
    
    return data_min_max_scaled

def split_datasets(data_min_max_scaled):
    """Split the initial dataset into training and testing datasets
    """
    # Seperate dataset for inputs and outputs
    X_data = data_min_max_scaled
    y_data = data["label"]

    # Create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.30, random_state=0)

    return X_train, X_test, y_train, y_test