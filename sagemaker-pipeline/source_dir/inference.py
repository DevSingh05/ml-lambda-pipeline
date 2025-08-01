import os
import joblib
import json
import numpy as np

def model_fn(model_dir):
    # Load the saved model
    return joblib.load(os.path.join(model_dir, "xgboost-model.joblib"))

def input_fn(request_body, content_type):
    # SageMaker passes the request body as a JSON string
    data = json.loads(request_body)
    return np.array(data)

def predict_fn(input_data, model):
    # Run model predictions
    return model.predict(input_data)

def output_fn(prediction, accept):
    # Return as JSON
    return json.dumps(prediction.tolist()), accept
