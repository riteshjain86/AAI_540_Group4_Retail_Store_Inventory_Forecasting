"""
SageMaker training job implementation for the retail demand forecasting model.

Uses the TensorFlow estimator (framework used by the LSTM/Transformer
candidates); `--model_type ridge` can be passed in hyperparameters to train
the classical baseline with the same entry point and container family.
"""

import os
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.inputs import TrainingInput
from sagemaker.tensorflow import TensorFlow

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerTrainingJob:
    """Manage SageMaker training jobs for the retail demand forecasting model."""

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

        logger.info(f"Initialized SageMaker training for region: {self.region}")

    def launch_training_job(self, train_s3_uri: str, val_s3_uri: str,
                            job_name: str = None, hyperparameters: dict = None,
                            instance_type: str = 'ml.m5.xlarge', test_s3_uri: str = None):
        """
        Launch a SageMaker training job.

        Args:
            train_s3_uri: S3 URI of the training Parquet split
            val_s3_uri: S3 URI of the validation Parquet split
            job_name: Name for the training job
            hyperparameters: Dictionary of hyperparameters (see scripts/train.py parse_args)
            instance_type: Instance type (CPU is sufficient at this dataset's scale;
                a GPU instance such as ml.g4dn.xlarge may reduce LSTM training time)
            test_s3_uri: Optional S3 URI of the test Parquet split; when given,
                the training script writes evaluation.json (test-set metrics
                vs a seasonal-naive baseline) into the model artifact

        Returns:
            Fitted TensorFlow estimator object
        """
        if job_name is None:
            job_name = f"retail-forecasting-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        if hyperparameters is None:
            hyperparameters = {
                'model_type': 'lstm',
                'sequence_length': 30,
                'lstm_units_1': 64,
                'lstm_units_2': 32,
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 0.001,
            }

        train_input = TrainingInput(s3_data=train_s3_uri, content_type='application/x-parquet')
        val_input = TrainingInput(s3_data=val_s3_uri, content_type='application/x-parquet')
        channels = {'train': train_input, 'validation': val_input}
        if test_s3_uri:
            channels['test'] = TrainingInput(s3_data=test_s3_uri, content_type='application/x-parquet')

        estimator = TensorFlow(
            entry_point='train.py',
            source_dir=str(self.scripts_dir),
            role=self.role_arn,
            instance_type=instance_type,
            instance_count=1,
            framework_version='2.13',
            py_version='py310',
            hyperparameters=hyperparameters,
            output_path=f"s3://{self.bucket_name}/{self.base_prefix}/models/",
            base_job_name='retail-forecasting',
            sagemaker_session=self.sagemaker_session,
            # Disable the TensorFlow estimator's automatic 'model_dir' hyperparameter
            # injection (an S3 URI for native TF checkpointing) - it collides with
            # train.py's own --model_dir argument and, if injected, overrides the
            # correct SM_MODEL_DIR (/opt/ml/model) with an S3 path, causing
            # model.joblib/model.keras to be saved to the wrong place. See
            # pipeline.py's TrainingStep for the full explanation (this bit the
            # SageMaker Pipeline path; applying it here too for consistency).
            model_dir=False,
        )

        logger.info(f"Launching training job: {job_name}")
        estimator.fit(channels, job_name=job_name)

        logger.info(f"Training job {job_name} completed")
        return estimator

    def get_training_job_status(self, job_name: str) -> dict:
        """
        Get the status of a training job.

        Args:
            job_name: Name of the training job

        Returns:
            Dictionary with job status information
        """
        response = self.sm_client.describe_training_job(TrainingJobName=job_name)
        return {
            'status': response['TrainingJobStatus'],
            'start_time': response.get('TrainingStartTime'),
            'end_time': response.get('TrainingEndTime'),
            'training_time': response.get('TrainingTimeInSeconds'),
            'billable_time': response.get('BillableTimeInSeconds'),
        }


def main():
    """
    Main function to launch a training job.

    By default trains the LSTM (the primary model per the design document).
    Set MODEL_TYPE=ridge to train the classical baseline instead (e.g. for
    the batch-transform deployment path, which only supports tabular models).
    """
    trainer = SageMakerTrainingJob()

    train_s3_uri = f"s3://{trainer.bucket_name}/{trainer.base_prefix}/train/train.parquet"
    val_s3_uri = f"s3://{trainer.bucket_name}/{trainer.base_prefix}/val/val.parquet"
    test_s3_uri = f"s3://{trainer.bucket_name}/{trainer.base_prefix}/test/test.parquet"

    model_type = os.environ.get('MODEL_TYPE', 'lstm')
    hyperparameters = None
    if model_type == 'ridge':
        hyperparameters = {
            'model_type': 'ridge',
            'ridge_alpha': float(os.environ.get('RIDGE_ALPHA', '1.0')),
        }

    estimator = trainer.launch_training_job(train_s3_uri, val_s3_uri, hyperparameters=hyperparameters,
                                            test_s3_uri=test_s3_uri)
    print(f"Training job launched: {estimator._current_job_name}")


if __name__ == "__main__":
    main()
