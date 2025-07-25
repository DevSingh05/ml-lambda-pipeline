import boto3

def lambda_handler(event, context):
    job_name = event["TrainingJobName"]
    sm_client = boto3.client("sagemaker")

    response = sm_client.describe_training_job(TrainingJobName=job_name)
    training_status = response["TrainingJobStatus"]

    return {
        "status": training_status
    }
