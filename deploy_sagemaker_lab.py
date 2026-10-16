"""
SageMaker Lab Environment Deployment Script.

This script is designed for SageMaker AI lab/Studio environments where
credentials are handled automatically via the attached execution role. It
sanity-checks the key AWS clients used by this project and prints the
remaining manual steps (each is its own module under src/aws/sagemaker/ so
they can be run and demoed independently).
"""

import logging
import os
import sys

import boto3
import sagemaker
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main deployment function for the SageMaker AI lab environment."""
    logger.info("=" * 60)
    logger.info("SageMaker AI Lab Environment Deployment Check")
    logger.info("=" * 60)
    logger.info("")

    load_dotenv()

    region = os.getenv('AWS_REGION', 'us-east-1')
    account_id = os.getenv('AWS_ACCOUNT_ID')
    role_arn = os.getenv('AWS_ROLE_ARN', f'arn:aws:iam::{account_id}:role/LabRole')

    logger.info(f"Region: {region}")
    logger.info(f"Account ID: {account_id}")
    logger.info(f"Role ARN: {role_arn}")
    logger.info("")

    try:
        sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=region))
        logger.info("[OK] SageMaker session initialized successfully")
    except Exception as e:
        logger.error(f"[FAIL] Failed to initialize SageMaker session: {e}")
        return False

    try:
        s3_client = boto3.client('s3', region_name=region)
        bucket_name = f"sagemaker-{region}-{account_id}"
        logger.info("[OK] S3 client initialized")
        logger.info(f"  Bucket: {bucket_name}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to initialize S3 client: {e}")
        return False

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"[OK] S3 bucket {bucket_name} exists")
    except Exception:
        logger.warning(f"[WARN] S3 bucket {bucket_name} does not exist yet; "
                       f"it will be created by data_upload.py")

    try:
        boto3.client('sagemaker', region_name=region)
        logger.info("[OK] SageMaker client initialized")
    except Exception as e:
        logger.error(f"[FAIL] Failed to initialize SageMaker client: {e}")
        return False

    try:
        boto3.client('cloudwatch', region_name=region)
        logger.info("[OK] CloudWatch client initialized")
    except Exception as e:
        logger.error(f"[FAIL] Failed to initialize CloudWatch client: {e}")
        return False

    logger.info("")
    logger.info("=" * 60)
    logger.info("Environment Check Complete")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps (each can be run and demoed independently):")
    logger.info("1. Upload data + processed splits to S3:")
    logger.info("   python -m src.aws.sagemaker.data_upload")
    logger.info("")
    logger.info("2. Create the Feature Store feature group:")
    logger.info("   python -m src.aws.sagemaker.feature_store")
    logger.info("")
    logger.info("3. Run SageMaker managed training (LSTM by default):")
    logger.info("   python -m src.aws.sagemaker.training")
    logger.info("")
    logger.info("4. Register the trained model in the Model Registry:")
    logger.info("   TRAINING_JOB_NAME=<job-name> python -m src.aws.sagemaker.model_registry")
    logger.info("")
    logger.info("5. Run a batch transform job (primary inference pattern):")
    logger.info("   python -m src.aws.sagemaker.batch_inference")
    logger.info("")
    logger.info("6. (Optional) Deploy a real-time endpoint:")
    logger.info("   python -m src.aws.sagemaker.deployment")
    logger.info("")
    logger.info("7. Set up model/data monitoring:")
    logger.info("   python -m src.aws.sagemaker.monitoring")
    logger.info("")
    logger.info("8. Create infrastructure monitoring dashboards:")
    logger.info("   python -m src.aws.sagemaker.infrastructure_monitoring")
    logger.info("")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
