# -----Imports-----
import pickle as pk

from PreProcess import *

# -----Prediction generation algorithm-----
def generate_prediction(hr_mean_raw, hr_std_raw):
    """Generate state of the body based on the heart rate mean and standard deviation
    """   
    # Min and max values for main features
    hr_mean_min_pre_norm = 50.777210
    hr_mean_max_pre_norm = 140.080622

    hr_std_min_pre_norm = 3.636572
    hr_std_max_pre_norm = 27.947662

    # Normalize inputs
    # z = (x - min)/(max - min)
    hr_mean_norm = (hr_mean_raw - hr_mean_min_pre_norm) / (hr_mean_max_pre_norm - hr_mean_min_pre_norm)
    hr_std_norm = (hr_std_raw - hr_std_min_pre_norm) / (hr_std_max_pre_norm - hr_std_min_pre_norm)

    # Final input
    final_input = [[hr_mean_norm, hr_std_norm]]

    # Load the model
    loaded_model = pk.load(open('./Model/voting_pickle_file', 'rb'))

    # Generate prediction
    result = loaded_model.predict(final_input)
    
    return str(result[0])

# # Get user input
# hr_mean_raw, hr_std_raw = process_hrv_data("./Data/hrv_data2.txt")
 
# Generate prediction
# result = generate_prediction(float(hr_mean_raw), float(hr_std_raw))
