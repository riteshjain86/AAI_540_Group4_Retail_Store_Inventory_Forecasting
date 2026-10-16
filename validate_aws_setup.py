"""
AWS SageMaker setup validation script.

This script validates that AWS credentials, permissions and local artifacts
are correctly in place before attempting to run the SageMaker workflow
(data upload -> feature store -> training -> registry -> batch inference ->
monitoring -> dashboards).
"""

import logging
import os
import sys

import boto3
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_env_file():
    """Validate that .env file exists and contains required variables."""
    logger.info("Validating .env file...")

    if not os.path.exists('.env'):
        logger.error(".env file not found. Please create it with your AWS credentials.")
        return False

    load_dotenv()

    required_vars = ['AWS_ACCOUNT_ID', 'AWS_REGION', 'AWS_ROLE_ARN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False

    logger.info(".env file validated successfully")
    logger.info(f"  Account ID: {os.getenv('AWS_ACCOUNT_ID')}")
    logger.info(f"  Region: {os.getenv('AWS_REGION')}")
    logger.info(f"  Role ARN: {os.getenv('AWS_ROLE_ARN')}")

    return True


def validate_aws_credentials():
    """Validate AWS credentials are correctly configured."""
    logger.info("Validating AWS credentials...")

    try:
        session = boto3.Session()
        credentials = session.get_credentials()

        if credentials is None:
            logger.error("AWS credentials not found. Please configure them.")
            return False

        logger.info("AWS credentials found")
        return True

    except Exception as e:
        logger.error(f"Error validating AWS credentials: {e}")
        return False


def validate_s3_access():
    """Validate S3 access and bucket creation."""
    logger.info("Validating S3 access...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        account_id = os.getenv('AWS_ACCOUNT_ID')
        bucket_name = f"sagemaker-{region}-{account_id}"

        s3_client = boto3.client('s3', region_name=region)

        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket {bucket_name} exists and is accessible")
        except Exception:
            logger.warning(f"S3 bucket {bucket_name} does not exist. It will be created automatically.")
            logger.info(f"Expected bucket name: {bucket_name}")

        return True

    except Exception as e:
        logger.error(f"Error validating S3 access: {e}")
        return False


def validate_sagemaker_role():
    """Validate SageMaker execution role exists and is accessible."""
    logger.info("Validating SageMaker role...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        role_arn = os.getenv('AWS_ROLE_ARN')

        iam_client = boto3.client('iam', region_name=region)
        role_name = role_arn.split('/')[-1]

        try:
            iam_client.get_role(RoleName=role_name)
            logger.info(f"SageMaker role {role_name} exists and is accessible")
            return True
        except iam_client.exceptions.NoSuchEntityException:
            logger.error(f"SageMaker role {role_name} does not exist")
            return False

    except Exception as e:
        # AWS Lab/Studio roles (e.g. "LabRole") are frequently not directly
        # describable by the user's own IAM permissions; treat this as a
        # warning rather than a hard failure.
        logger.warning(f"Could not independently verify the role via IAM (expected in some lab "
                       f"accounts): {e}")
        return True


def validate_sagemaker_permissions():
    """Validate that the caller has necessary SageMaker permissions."""
    logger.info("Validating SageMaker permissions...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        sagemaker_client.list_training_jobs(MaxResults=1)

        logger.info("SageMaker permissions validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating SageMaker permissions: {e}")
        logger.error("Ensure your IAM role/user has SageMakerFullAccess or equivalent permissions")
        return False


def validate_cloudwatch_permissions():
    """Validate CloudWatch permissions for monitoring."""
    logger.info("Validating CloudWatch permissions...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        cloudwatch = boto3.client('cloudwatch', region_name=region)
        cloudwatch.list_dashboards()

        logger.info("CloudWatch permissions validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating CloudWatch permissions: {e}")
        logger.error("Ensure your IAM role/user has CloudWatchFullAccess or equivalent permissions")
        return False


def validate_python_dependencies():
    """Validate that required Python packages are installed, including the
    SageMaker Python SDK v2 pin this project relies on."""
    logger.info("Validating Python dependencies...")

    required_packages = ['boto3', 'sagemaker', 'pandas', 'numpy', 'sklearn', 'mlflow', 'tensorflow']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Run: pip install -r requirements.txt")
        return False

    import sagemaker
    if not sagemaker.__version__.startswith('2.'):
        logger.error(f"sagemaker=={sagemaker.__version__} found; this project targets "
                     f"SageMaker SDK v2 (the AWS Lab environment does not support v3). "
                     f"Run: pip install 'sagemaker>=2.190.0,<3.0.0'")
        return False

    logger.info(f"All required Python packages are installed (sagemaker=={sagemaker.__version__})")
    return True


def validate_data_files():
    """Validate that the raw dataset is present."""
    logger.info("Validating data files...")

    required_files = ['data/raw/retail_store_inventory.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        logger.error("Download the Kaggle 'Retail Store Inventory Forecasting' dataset and place "
                     "retail_store_inventory.csv under data/raw/")
        return False

    logger.info("All required data files exist")
    return True


def run_validation():
    """Run all validation checks."""
    logger.info("=" * 60)
    logger.info("AWS SageMaker Setup Validation")
    logger.info("=" * 60)
    logger.info("")

    checks = [
        ("Environment File", validate_env_file),
        ("AWS Credentials", validate_aws_credentials),
        ("S3 Access", validate_s3_access),
        ("SageMaker Role", validate_sagemaker_role),
        ("SageMaker Permissions", validate_sagemaker_permissions),
        ("CloudWatch Permissions", validate_cloudwatch_permissions),
        ("Python Dependencies", validate_python_dependencies),
        ("Data Files", validate_data_files),
    ]

    results = []
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        results.append((check_name, check_func()))

    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"[{status}] {check_name}")

    logger.info("")
    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.info("All validations passed! You can proceed with the SageMaker workflow.")
        return True

    logger.error("Some validations failed. Please fix the issues above before deploying.")
    return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
