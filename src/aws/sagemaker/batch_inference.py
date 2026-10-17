"""
SageMaker batch transform job implementation for batch inference.

This is the primary deployment pattern for the project: retail demand
forecasts are generated on a schedule (e.g. nightly) rather than per
individual request, so a batch transform job over a prepared feature file is
more appropriate than a standing real-time endpoint.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import boto3
import pandas as pd
import sagemaker
from dotenv import load_dotenv
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.transformer import Transformer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerBatchInference:
    """Manage SageMaker batch transform jobs for batch inference."""

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

        logger.info(f"Initialized SageMaker batch inference for region: {self.region}")

    def create_transformer(self, model_s3_uri: str, instance_type: str = 'ml.m5.xlarge',
                           instance_count: int = 1) -> Transformer:
        """
        Create a Transformer for the classical (tabular) model artifact.

        Batch transform over the CSV/tabular `batch_transform.py` entry point
        is only supported for the classical scikit-learn models; a sequence
        (LSTM/Transformer) model needs pre-built windows and is better served
        through the real-time endpoint in deployment.py for ad hoc scoring.

        Args:
            model_s3_uri: S3 URI of the trained classical model.tar.gz
            instance_type: Instance type for the transform job
            instance_count: Number of instances

        Returns:
            Transformer object
        """
        model = SKLearnModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            entry_point='batch_transform.py',
            source_dir=str(self.scripts_dir),
            framework_version='1.2-1',
            py_version='py3',
            sagemaker_session=self.sagemaker_session,
        )

        transformer = model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=f"s3://{self.bucket_name}/{self.base_prefix}/predictions/",
            sagemaker_session=self.sagemaker_session,
        )

        logger.info("Transformer created")
        return transformer

    def run_batch_transform(self, transformer: Transformer, input_s3_uri: str,
                            job_name: str = None, content_type: str = 'text/csv',
                            split_type: str = 'Line'):
        """
        Run a batch transform job.

        Args:
            transformer: Transformer object
            input_s3_uri: S3 URI of the input feature file(s)
            job_name: Name for the transform job
            content_type: Content type of input data
            split_type: How to split input data (Line, Record, etc.)

        Returns:
            Transform job name
        """
        if job_name is None:
            job_name = f"retail-forecasting-batch-transform-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        logger.info(f"Starting batch transform job: {job_name}")
        transformer.transform(
            data=input_s3_uri, content_type=content_type, split_type=split_type,
            job_name=job_name, wait=False,
        )

        logger.info(f"Batch transform job {job_name} started")
        return job_name

    def prepare_batch_input(self, data_path: str = 'data/processed/test.parquet',
                            local_csv: str = 'batch_input.csv') -> str:
        """
        Engineer features from a Parquet split and upload them as the
        header-less CSV batch-transform input expected by
        scripts/batch_transform.py's classical (tabular) model.

        Args:
            data_path: Local path to a raw Parquet split (e.g. the test set)
            local_csv: Local path to write the intermediate feature CSV to

        Returns:
            S3 URI (prefix) where the batch input file was uploaded
        """
        import sys
        sys.path.insert(0, str(self.scripts_dir))
        from train import engineer_features, CATEGORICAL_COLS

        df = pd.read_parquet(data_path)
        engineered, feature_cols = engineer_features(df)
        numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
        engineered[numeric_cols + CATEGORICAL_COLS].to_csv(local_csv, index=False, header=False)

        s3_prefix = f"{self.base_prefix}/predictions/batch-input"
        s3_key = f"{s3_prefix}/{Path(local_csv).name}"
        boto3.client('s3', region_name=self.region).upload_file(local_csv, self.bucket_name, s3_key)

        s3_uri = f"s3://{self.bucket_name}/{s3_prefix}/"
        logger.info(f"Uploaded batch input ({len(engineered)} rows) to {s3_uri}")
        return s3_uri

    def get_transform_job_status(self, job_name: str) -> dict:
        """
        Get the status of a transform job.

        Args:
            job_name: Name of the transform job

        Returns:
            Dictionary with job status information
        """
        response = self.sm_client.describe_transform_job(TransformJobName=job_name)
        return {
            'status': response['TransformJobStatus'],
            'start_time': response.get('TransformStartTime'),
            'end_time': response.get('TransformEndTime'),
            'instance_type': response.get('TransformResources', {}).get('InstanceType'),
            'instance_count': response.get('TransformResources', {}).get('InstanceCount'),
        }

    def wait_for_transform_job(self, job_name: str, poll_interval: int = 30):
        """
        Wait for a transform job to complete.

        Args:
            job_name: Name of the transform job
            poll_interval: Polling interval in seconds
        """
        logger.info(f"Waiting for transform job {job_name} to complete...")

        status = None
        while True:
            status_info = self.get_transform_job_status(job_name)
            status = status_info['status']
            logger.info(f"Transform job status: {status}")
            if status in ('Completed', 'Failed', 'Stopped'):
                break
            time.sleep(poll_interval)

        logger.info(f"Transform job {job_name} finished with status: {status}")
        return status_info

    def list_transform_jobs(self) -> list:
        """List all transform jobs."""
        response = self.sm_client.list_transform_jobs()
        return [job['TransformJobName'] for job in response['TransformJobSummaries']]


def main():
    """Main function to run batch inference.

    Requires the TRAINING_JOB_NAME environment variable to be set to a
    SageMaker training job that produced a classical (tabular) model
    artifact, e.g. one launched with hyperparameters={'model_type': 'ridge'}
    (see training.py). Batch transform does not support the sequence
    (LSTM/Transformer) models; use deployment.py's real-time endpoint for
    those instead.
    """
    training_job_name = os.environ.get('TRAINING_JOB_NAME')
    if not training_job_name:
        raise ValueError(
            "Set TRAINING_JOB_NAME to the SageMaker training job whose classical "
            "model.tar.gz should be used for batch inference."
        )

    batch_inference = SageMakerBatchInference()

    model_s3_uri = (f"s3://{batch_inference.bucket_name}/{batch_inference.base_prefix}/"
                    f"models/{training_job_name}/output/model.tar.gz")
    input_s3_uri = batch_inference.prepare_batch_input()

    transformer = batch_inference.create_transformer(model_s3_uri)
    transform_job_name = batch_inference.run_batch_transform(transformer, input_s3_uri)
    print(f"Batch transform job started: {transform_job_name}")

    status_info = batch_inference.wait_for_transform_job(transform_job_name)
    print(f"Transform job completed: {status_info}")


if __name__ == "__main__":
    main()
