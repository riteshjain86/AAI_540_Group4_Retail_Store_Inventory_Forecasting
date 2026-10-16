"""
SageMaker Model Monitor, forecast-accuracy metrics and CloudWatch dashboard
implementation for the retail demand forecasting system.
"""

import json
import os
from datetime import datetime

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CUSTOM_METRIC_NAMESPACE = 'RetailForecasting/ModelQuality'


class SageMakerModelMonitor:
    """Manage SageMaker Model Monitor, custom forecast-accuracy metrics and CloudWatch dashboards."""

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
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)

        logger.info(f"Initialized Model Monitor for region: {self.region}")

    def create_baseline(self, baseline_data_s3_uri: str, output_s3_uri: str = None,
                        monitor_name: str = "retail-forecasting-monitor"):
        """
        Create a data-quality baseline from the training split (schema, ranges,
        missing-value rates, categorical value sets).

        Args:
            baseline_data_s3_uri: S3 URI of baseline (training) data
            output_s3_uri: S3 URI for baseline statistics output
            monitor_name: Name of the monitor

        Returns:
            DefaultModelMonitor object
        """
        if output_s3_uri is None:
            output_s3_uri = f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/baseline/"

        logger.info(f"Creating baseline with data from {baseline_data_s3_uri}")

        my_monitor = DefaultModelMonitor(
            role=self.role_arn, instance_count=1, instance_type='ml.m5.xlarge',
            volume_size_in_gb=20, max_runtime_in_seconds=1800,
            sagemaker_session=self.sagemaker_session,
        )

        baseline_job = my_monitor.suggest_baseline(
            baseline_dataset=baseline_data_s3_uri,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=output_s3_uri,
        )

        logger.info(f"Baseline job started: {baseline_job}")
        return my_monitor

    def create_monitoring_schedule(self, endpoint_name: str, baseline_s3_uri: str,
                                   output_s3_uri: str = None,
                                   monitor_name: str = "retail-forecasting-monitor",
                                   schedule_cron_expression: str = "cron(0 * ? * * *)"):
        """
        Create a data-quality monitoring schedule for a real-time endpoint.

        Args:
            endpoint_name: Name of the endpoint to monitor
            baseline_s3_uri: S3 URI of baseline statistics
            output_s3_uri: S3 URI for monitoring output
            monitor_name: Name of the monitor
            schedule_cron_expression: Cron expression for the monitoring schedule

        Returns:
            DefaultModelMonitor object
        """
        if output_s3_uri is None:
            output_s3_uri = f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/results/"

        logger.info(f"Creating monitoring schedule for endpoint: {endpoint_name}")

        my_monitor = DefaultModelMonitor(
            role=self.role_arn, instance_count=1, instance_type='ml.m5.xlarge',
            volume_size_in_gb=20, max_runtime_in_seconds=1800,
            sagemaker_session=self.sagemaker_session,
        )

        my_monitor.create_monitoring_schedule(
            endpoint_input=endpoint_name,
            output_s3_uri=output_s3_uri,
            statistics=f"{baseline_s3_uri}/statistics.json",
            constraints=f"{baseline_s3_uri}/constraints.json",
            schedule_cron_expression=schedule_cron_expression,
            monitor_schedule_name=monitor_name,
        )

        logger.info(f"Monitoring schedule created: {monitor_name}")
        return my_monitor

    def publish_forecast_accuracy_metrics(self, metrics: dict, segment: str = 'overall'):
        """
        Publish MAE / RMSE / WAPE as custom CloudWatch metrics once actual
        demand becomes available for a batch of forecasts.

        This is the "notebook-based monitoring report" supplement the design
        document allows where managed SageMaker Model Monitor ground-truth
        merging is unavailable in the course lab.

        Args:
            metrics: Dict with any of 'mae', 'rmse', 'wape', 'smape'
            segment: Label for the slice these metrics apply to (e.g. a
                Category, Store ID, or 'overall')
        """
        metric_data = [
            {
                'MetricName': name,
                'Dimensions': [{'Name': 'Segment', 'Value': segment}],
                'Timestamp': datetime.utcnow(),
                'Value': float(value),
            }
            for name, value in metrics.items() if value == value  # filters out NaN
        ]
        if not metric_data:
            return
        self.cloudwatch.put_metric_data(Namespace=CUSTOM_METRIC_NAMESPACE, MetricData=metric_data)
        logger.info(f"Published forecast accuracy metrics for segment '{segment}': {metrics}")

    def create_cloudwatch_dashboard(self, dashboard_name: str = "retail-forecasting-dashboard"):
        """
        Create a CloudWatch dashboard summarising endpoint health and forecast accuracy.

        Args:
            dashboard_name: Name of the dashboard

        Returns:
            Dashboard configuration
        """
        endpoint_dim = f"{self.base_prefix}-endpoint"
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "ModelLatency", "EndpointName", endpoint_dim],
                            [".", "Invocations", ".", ".", {"stat": "Sum"}],
                            [".", "4XXError", ".", ".", {"stat": "Sum"}],
                            [".", "5XXError", ".", ".", {"stat": "Sum"}],
                        ],
                        "period": 300, "stat": "Average", "region": self.region, "title": "Endpoint Metrics",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            [CUSTOM_METRIC_NAMESPACE, "mae", "Segment", "overall"],
                            [".", "rmse", ".", "."],
                        ],
                        "period": 86400, "stat": "Average", "region": self.region,
                        "title": "Forecast Accuracy (MAE / RMSE)",
                    },
                },
                {
                    "type": "metric", "x": 0, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [[CUSTOM_METRIC_NAMESPACE, "wape", "Segment", "overall"]],
                        "period": 86400, "stat": "Average", "region": self.region,
                        "title": "Forecast Accuracy (WAPE)",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "EndpointName", endpoint_dim],
                            [".", "MemoryUtilization", ".", "."],
                        ],
                        "period": 300, "stat": "Average", "region": self.region, "title": "Infrastructure Metrics",
                    },
                },
                {
                    "type": "log", "x": 0, "y": 12, "width": 24, "height": 6,
                    "properties": {
                        "logGroupName": f"/aws/sagemaker/Endpoints/{endpoint_dim}",
                        "region": self.region, "title": "Endpoint Logs",
                    },
                },
            ]
        }

        try:
            self.cloudwatch.put_dashboard(DashboardName=dashboard_name, DashboardBody=json.dumps(dashboard_body))
            logger.info(f"CloudWatch dashboard {dashboard_name} created")
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")

        return dashboard_body

    def describe_monitoring_schedule(self, monitor_name: str) -> dict:
        """
        Get information about a monitoring schedule.

        Args:
            monitor_name: Name of the monitoring schedule

        Returns:
            Dictionary with monitoring schedule information
        """
        response = self.sm_client.describe_monitoring_schedule(MonitoringScheduleName=monitor_name)
        return {
            'monitoring_schedule_name': response['MonitoringScheduleName'],
            'monitoring_schedule_status': response['MonitoringScheduleStatus'],
            'monitoring_type': response.get('MonitoringType'),
            'endpoint_name': response.get('EndpointName'),
            'schedule': response.get('ScheduleConfig'),
        }

    def stop_monitoring_schedule(self, monitor_name: str):
        """Stop a monitoring schedule."""
        logger.info(f"Stopping monitoring schedule: {monitor_name}")
        self.sm_client.stop_monitoring_schedule(MonitoringScheduleName=monitor_name)
        logger.info("Monitoring schedule stopped")

    def delete_monitoring_schedule(self, monitor_name: str):
        """Delete a monitoring schedule."""
        logger.info(f"Deleting monitoring schedule: {monitor_name}")
        self.sm_client.delete_monitoring_schedule(MonitoringScheduleName=monitor_name)
        logger.info("Monitoring schedule deleted")


def main():
    """Main function to demonstrate monitoring setup."""
    monitor = SageMakerModelMonitor()
    dashboard_body = monitor.create_cloudwatch_dashboard()

    print("CloudWatch dashboard created")
    print(f"Dashboard widgets: {len(dashboard_body['widgets'])}")

    # Example: publish accuracy metrics once ground truth is known
    # monitor.publish_forecast_accuracy_metrics({'mae': 87.9, 'rmse': 108.3, 'wape': 0.64})


if __name__ == "__main__":
    main()
