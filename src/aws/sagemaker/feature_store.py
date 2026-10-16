"""
SageMaker Feature Store implementation for the retail store inventory dataset.
"""

import io
import logging
import os
import time
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetailInventoryFeatureStore:
    """Manage SageMaker Feature Store for the retail store inventory data."""

    def __init__(self):
        """Initialize SageMaker session and feature store configuration."""
        load_dotenv()

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.role_arn = os.getenv('AWS_ROLE_ARN')
        if not self.role_arn:
            raise ValueError('Set AWS_ROLE_ARN to your SageMaker execution role ARN.')
        account_id = os.getenv('AWS_ACCOUNT_ID')
        if not account_id:
            account_id = boto3.client('sts', region_name=self.region).get_caller_identity()['Account']
        self.bucket_name = f"sagemaker-{self.region}-{account_id}"
        self.base_prefix = "aai540/retail-forecasting"

        self.sagemaker_session = Session(boto3.session.Session(region_name=self.region))

        # Feature group names
        self.feature_group_name = "retail-inventory-features-raw"
        self.feature_group_name_processed = "retail-inventory-features-processed"

        logger.info(f"Initialized Feature Store for region: {self.region}")

    def _get_feature_definitions(self, df: pd.DataFrame) -> list:
        """
        Generate feature definitions from DataFrame schema.

        Args:
            df: DataFrame to extract schema from

        Returns:
            List of feature definitions for Feature Group
        """
        feature_definitions = []

        for column_name, dtype in df.dtypes.items():
            if dtype == 'object':
                feature_type = 'String'
            elif dtype in ['int64', 'int32']:
                feature_type = 'Integral'
            elif dtype in ['float64', 'float32']:
                feature_type = 'Fractional'
            else:
                feature_type = 'String'

            feature_definitions.append({
                'FeatureName': column_name,
                'FeatureType': feature_type
            })

        return feature_definitions

    def create_feature_group(self, df: pd.DataFrame, feature_group_name: str,
                             record_identifier_name: str = 'record_id',
                             event_time_feature_name: str = 'event_time') -> FeatureGroup:
        """
        Create a SageMaker Feature Group.

        Args:
            df: Sample DataFrame to define schema
            feature_group_name: Name of the feature group
            record_identifier_name: Name of the record identifier column
            event_time_feature_name: Name of the event time column

        Returns:
            FeatureGroup object
        """
        for column in (record_identifier_name, event_time_feature_name):
            if column not in df.columns:
                raise ValueError(f'Missing required feature column: {column}')

        feature_group = FeatureGroup(
            name=feature_group_name,
            sagemaker_session=self.sagemaker_session,
        )
        feature_group.load_feature_definitions(data_frame=df)

        # Only a missing group is a reason to create a new one.
        try:
            existing = self.sagemaker_session.sagemaker_client.describe_feature_group(
                FeatureGroupName=feature_group_name
            )
        except ClientError as exc:
            if exc.response['Error']['Code'] not in ('ResourceNotFound', 'ResourceNotFoundException'):
                raise
        else:
            if existing.get('FeatureGroupStatus') != 'Created':
                raise RuntimeError(f"Existing group is not ready: {existing.get('FeatureGroupStatus')}")
            expected = {item.to_dict()['FeatureName']: item.to_dict()['FeatureType']
                        for item in feature_group.feature_definitions}
            actual = {item['FeatureName']: item['FeatureType']
                      for item in existing['FeatureDefinitions']}
            if any(actual.get(name) != kind for name, kind in expected.items()):
                raise ValueError('Existing feature group schema does not match the input data.')
            logger.info(f'Feature group {feature_group_name} already exists')
            return feature_group

        # Create feature group
        logger.info(f"Creating feature group: {feature_group_name}")
        feature_group.create(
            s3_uri=f"s3://{self.bucket_name}/{self.base_prefix}/feature-store/",
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=self.role_arn
        )

        # Wait for feature group to be created
        logger.info("Waiting for feature group to be created...")
        status = feature_group.describe().get('FeatureGroupStatus')
        deadline = time.monotonic() + 900
        while status == 'Creating':
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Feature group creation timed out: {feature_group_name}')
            time.sleep(5)
            status = feature_group.describe().get('FeatureGroupStatus')
            logger.info(f"Feature group status: {status}")

        if status != 'Created':
            raise RuntimeError(f"Failed to create feature group. Status: {status}")

        logger.info(f"Feature group {feature_group_name} created successfully")
        return feature_group

    def ingest_data(self, feature_group: FeatureGroup, df: pd.DataFrame,
                    max_workers: int = 2, wait: bool = True) -> None:
        """
        Ingest data into a Feature Group.

        Args:
            feature_group: FeatureGroup object
            df: DataFrame with data to ingest (must already contain record_id / event_time)
            max_workers: Number of parallel workers
            wait: Whether to wait for ingestion to complete
        """
        logger.info(f"Ingesting {len(df)} records into feature group...")
        feature_group.ingest(data_frame=df, max_workers=max_workers, wait=wait)
        logger.info("Data ingestion completed")

    @staticmethod
    def _add_record_identifiers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the record_id (composite key) and event_time (Unix seconds) columns
        required by Feature Store, since the dataset's natural key is
        (Date, Store ID, Product ID) rather than a single column.
        """
        df = df.copy()
        df['record_id'] = (df['Date'].astype(str) + '#' + df['Store ID'].astype(str)
                           + '#' + df['Product ID'].astype(str))
        df['event_time'] = pd.to_datetime(df['Date']).astype('int64') / 1e9
        df['Date'] = df['Date'].astype(str)
        return df

    def create_retail_feature_group(self, data_path: str = None):
        """
        Create and populate the raw-feature Feature Group for the retail dataset.

        Args:
            data_path: S3 URI of the raw CSV (defaults to the project's raw prefix)

        Returns:
            Dictionary with the created feature group(s)
        """
        uri = data_path or f's3://{self.bucket_name}/{self.base_prefix}/raw/retail_store_inventory.csv'
        parsed = urlparse(uri)
        if parsed.scheme != 's3' or not parsed.netloc or not parsed.path.lstrip('/'):
            raise ValueError('data_path must be a valid s3://bucket/key URI')
        response = boto3.client('s3', region_name=self.region).get_object(
            Bucket=parsed.netloc, Key=parsed.path.lstrip('/')
        )
        body = response['Body']
        try:
            df = pd.read_csv(io.BytesIO(body.read()))
        finally:
            body.close()
        if df.empty:
            raise ValueError('Raw S3 dataset is empty')

        df = self._add_record_identifiers(df)
        logger.info(f'Loaded {len(df)} rows from {uri}')

        logger.info("Creating raw feature group...")
        raw_fg = self.create_feature_group(df, self.feature_group_name)
        self.ingest_data(raw_fg, df)
        logger.info("Raw feature group created and populated")

        return {'raw': raw_fg, 'processed': None}

    def get_feature_group(self, feature_group_name: str) -> FeatureGroup:
        """
        Get an existing Feature Group.

        Args:
            feature_group_name: Name of the feature group

        Returns:
            FeatureGroup object
        """
        return FeatureGroup(name=feature_group_name, sagemaker_session=self.sagemaker_session)

    def retrieve_features(self, feature_group_name: str, record_identifiers: list) -> pd.DataFrame:
        """
        Retrieve features from the Feature Store.

        Args:
            feature_group_name: Name of the feature group
            record_identifiers: List of record identifiers to retrieve

        Returns:
            DataFrame with retrieved features
        """
        feature_group = self.get_feature_group(feature_group_name)
        feature_group_query = feature_group.create_query(feature_group_name=feature_group_name)
        logger.info(f"Retrieving features for {len(record_identifiers)} records")
        return feature_group_query


def main():
    """Main function to create and populate feature groups."""
    feature_store = RetailInventoryFeatureStore()
    feature_groups = feature_store.create_retail_feature_group()

    print("\n=== Feature Groups Created ===")
    for name, fg in feature_groups.items():
        if fg:
            print(f"{name}: {fg.name}")
            print(f"Status: {fg.describe().get('FeatureGroupStatus')}")


if __name__ == "__main__":
    main()
