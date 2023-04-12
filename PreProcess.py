# -----Imports-----
import numpy as np

# # -----Processing algorithms-----
def process_hrv_data(data):
    """Generate heart rate mean and standard deviation from heart rate variability data
    """    
    # Load HRV data
    hrv_data = np.array(data)

    # Calculate RR intervals from HRV data
    rr_intervals = 1000 / hrv_data

    # Calculate heart rate from RR intervals
    hr = 60 / rr_intervals

    # Calculate HR mean and HR standard deviation
    hr_mean = np.mean(hr)
    hr_std = np.std(hr)

    return hr_mean, hr_std
