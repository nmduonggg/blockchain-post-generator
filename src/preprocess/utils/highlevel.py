import numpy as np
from pmdarima import auto_arima

def arima_pipeline(timestamps, values, n_predictions=500):
    
    # interpolate values
    timestamps = np.array(timestamps)
    new_timestamps = np.linspace(int(min(timestamps)), int(max(timestamps))+1, 1000).astype(int)
    new_values = np.interp(new_timestamps, timestamps, values)
    
    model = auto_arima(new_values, 
                       start_p=100, max_p=500,
                       start_q=100, max_q=500,
                       seasonal=False,
                       stepwise=False, trace=False,
                       random=True, n_jobs=32, maxiter=50,
                       error_action='ignore')
    forecast = model.predict(n_predictions)
    
    additional_timestamps = (new_timestamps[-1] - new_timestamps[-2]) * np.arange(1, n_predictions+1) + timestamps[-1]
    new_timestamps = new_timestamps.tolist() + additional_timestamps.tolist()
    new_values = new_values.tolist() + forecast.tolist()
    
    return n_predictions, new_timestamps, new_values
    
    