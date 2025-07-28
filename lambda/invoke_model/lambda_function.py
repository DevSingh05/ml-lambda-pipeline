import boto3
import json

runtime = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = 'xgboost-endpoint-v2'

def lambda_handler(event, context):
    features = event.get("features")
    if not features:
        return {"error": "Missing 'features' in input"}

    payload = ",".join(map(str, features))

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=payload
    )

    result = response['Body'].read().decode('utf-8')
    return {
        "input": features,
        "prediction": result.strip()
    }
