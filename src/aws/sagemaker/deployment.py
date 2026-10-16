"""
SageMaker model deployment and endpoint management.

Per the design document, batch inference (see batch_inference.py) is the
preferred deployment pattern for scheduled demand forecasts; a real-time
endpoint is an optional extension kept here for completeness / future use.
"""

import os
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.tensorflow import TensorFlowModel

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerDeployment:
    """Manage SageMaker model deployment and endpoints."""

    def __init__(self):
        """Initialize SageMaker session and configuration."""
        load_dotenv()

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.role_arn = os.getenv('AWS_ROLE_ARN')
        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.bucket_name = f"sagemaker-{self.region}-{self.account_id}"
        self.base_prefix = "aai540/retail-forecasting"

        self.sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=self.region))
        self.sm_client = boto3.client('sagemaker', region_name=self.region)
        self.scripts_dir = Path(__file__).parent / 'scripts'

        logger.info(f"Initialized SageMaker deployment for region: {self.region}")

    def deploy_model(self, model_s3_uri: str, endpoint_name: str = None,
                     instance_type: str = 'ml.m5.xlarge',
                     initial_instance_count: int = 1,
                     enable_data_capture: bool = True):
        """
        Deploy a trained LSTM/Transformer model to a SageMaker real-time endpoint.

        Args:
            model_s3_uri: S3 URI of the trained model.tar.gz
            endpoint_name: Name for the endpoint
            instance_type: Instance type for the endpoint
            initial_instance_count: Number of instances
            enable_data_capture: Whether to enable data capture for monitoring

        Returns:
            Predictor object for the endpoint
        """
        if endpoint_name is None:
            endpoint_name = f"retail-forecasting-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        model = TensorFlowModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            entry_point='inference.py',
            source_dir=str(self.scripts_dir),
            framework_version='2.13',
            sagemaker_session=self.sagemaker_session,
        )

        data_capture_config = None
        if enable_data_capture:
            capture_s3_uri = f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/data-capture/"
            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri=capture_s3_uri,
                capture_options=['REQUEST', 'RESPONSE'],
                json_content_types=['application/json'],
            )

        logger.info(f"Deploying model to endpoint: {endpoint_name}")
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            data_capture_config=data_capture_config,
        )

        logger.info(f"Model deployed to endpoint: {endpoint_name}")
        return predictor

    def invoke_endpoint(self, endpoint_name: str, sequence: list, content_type: str = 'application/json'):
        """
        Invoke a SageMaker endpoint for a single forecast.

        Args:
            endpoint_name: Name of the endpoint
            sequence: A (sequence_length, n_features) list of lists for one store/product series
            content_type: Content type of the request

        Returns:
            Prediction results
        """
        import json

        runtime_client = boto3.client('runtime.sagemaker', region_name=self.region)
        payload = json.dumps({'sequence': sequence})
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=content_type, Body=payload,
        )
        return response['Body'].read().decode('utf-8')

    def delete_endpoint(self, endpoint_name: str):
        """Delete a SageMaker endpoint. Always do this when a demo/lab endpoint is no longer needed."""
        logger.info(f"Deleting endpoint: {endpoint_name}")
        self.sm_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint {endpoint_name} deleted")

    def describe_endpoint(self, endpoint_name: str) -> dict:
        """
        Get endpoint information.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Dictionary with endpoint information
        """
        response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        return {
            'endpoint_name': response['EndpointName'],
            'endpoint_status': response['EndpointStatus'],
            'creation_time': response.get('CreationTime'),
            'endpoint_config_name': response.get('EndpointConfigName'),
            'instance_type': response.get('ProductionVariants', [{}])[0].get('InstanceType'),
            'instance_count': response.get('ProductionVariants', [{}])[0].get('InitialInstanceCount'),
        }

    def list_endpoints(self) -> list:
        """List all SageMaker endpoints."""
        response = self.sm_client.list_endpoints()
        return [ep['EndpointName'] for ep in response['Endpoints']]


def main():
    """Main function to deploy model."""
    deployer = SageMakerDeployment()

    model_s3_uri = (
        f"s3://{deployer.bucket_name}/{deployer.base_prefix}/models/"
        "<training-job-name>/output/model.tar.gz"
    )

    predictor = deployer.deploy_model(model_s3_uri)
    print(f"Model deployed. Endpoint: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
