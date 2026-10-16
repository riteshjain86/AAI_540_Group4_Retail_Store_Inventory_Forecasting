"""
Infrastructure monitoring dashboards for the retail demand forecasting ML system.
"""

import json
import os

import boto3
from dotenv import load_dotenv

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfrastructureMonitoring:
    """Create and manage CloudWatch dashboards for infrastructure monitoring."""

    def __init__(self):
        """Initialize AWS clients and configuration."""
        load_dotenv()

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.base_prefix = "aai540/retail-forecasting"

        self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        self.sagemaker = boto3.client('sagemaker', region_name=self.region)

        logger.info(f"Initialized infrastructure monitoring for region: {self.region}")

    def create_comprehensive_dashboard(self, dashboard_name: str = "ml-system-monitoring"):
        """
        Create a comprehensive CloudWatch dashboard for the ML system
        (training jobs, endpoint, batch transform, S3, Feature Store, model quality).

        Args:
            dashboard_name: Name of the dashboard

        Returns:
            Dashboard configuration
        """
        endpoint_dim = f"{self.base_prefix}-endpoint"
        dashboard_body = {
            "widgets": [
                # Row 1: SageMaker Training Jobs
                {
                    "type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "TrainingJobName", {"label": "Training CPU"}],
                            [".", "MemoryUtilization", ".", {"label": "Training Memory"}],
                            [".", "GPUUtilization", ".", {"label": "Training GPU"}],
                        ],
                        "period": 300, "stat": "Average", "region": self.region,
                        "title": "SageMaker Training Jobs", "stacked": False,
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "BillableTimeInSeconds", "TrainingJobName", {"stat": "Sum"}],
                            [".", "TrainingTimeInSeconds", ".", {"stat": "Sum"}],
                        ],
                        "period": 86400, "stat": "Sum", "region": self.region, "title": "Training Job Duration",
                    },
                },
                # Row 2: SageMaker Endpoint
                {
                    "type": "metric", "x": 0, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "ModelLatency", "EndpointName", endpoint_dim],
                            [".", "OverheadLatency", ".", "."],
                            [".", "Invocations", ".", {"stat": "Sum"}],
                        ],
                        "period": 300, "stat": "Average", "region": self.region,
                        "title": "Endpoint Latency & Invocations",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "4XXError", "EndpointName", endpoint_dim, {"stat": "Sum"}],
                            [".", "5XXError", ".", ".", {"stat": "Sum"}],
                            [".", "Invocations", ".", ".", {"stat": "Sum"}],
                        ],
                        "period": 300, "stat": "Sum", "region": self.region, "title": "Endpoint Error Rates",
                    },
                },
                # Row 3: Instance Metrics
                {
                    "type": "metric", "x": 0, "y": 12, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/EC2", "CPUUtilization", "InstanceId", "*", {"label": "EC2 CPU"}],
                            [".", "DiskReadBytes", ".", "*", {"label": "Disk Read"}],
                            [".", "DiskWriteBytes", ".", "*", {"label": "Disk Write"}],
                        ],
                        "period": 300, "stat": "Average", "region": self.region,
                        "title": "Instance Resource Utilization",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 12, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/EC2", "NetworkIn", "InstanceId", "*", {"stat": "Sum"}],
                            [".", "NetworkOut", ".", "*", {"stat": "Sum"}],
                        ],
                        "period": 300, "stat": "Sum", "region": self.region, "title": "Network Traffic",
                    },
                },
                # Row 4: S3 Operations
                {
                    "type": "metric", "x": 0, "y": 18, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/S3", "BucketSizeBytes", "BucketName", f"{self.base_prefix}"],
                            [".", "NumberOfObjects", ".", "."],
                        ],
                        "period": 86400, "stat": "Average", "region": self.region, "title": "S3 Storage",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 18, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/S3", "4xxErrors", "BucketName", f"{self.base_prefix}", {"stat": "Sum"}],
                            [".", "5xxErrors", ".", ".", {"stat": "Sum"}],
                            [".", "GetRequests", ".", ".", {"stat": "Sum"}],
                            [".", "PutRequests", ".", ".", {"stat": "Sum"}],
                        ],
                        "period": 300, "stat": "Sum", "region": self.region, "title": "S3 Request Metrics",
                    },
                },
                # Row 5: Feature Store
                {
                    "type": "metric", "x": 0, "y": 24, "width": 24, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "GetRecordsLatency", "FeatureGroupName", "retail-inventory-features-raw"],
                            [".", "PutRecordLatency", ".", "."],
                            [".", "GetRecordsSuccess", ".", {"stat": "Sum"}],
                            [".", "PutRecordSuccess", ".", {"stat": "Sum"}],
                        ],
                        "period": 300, "stat": "Average", "region": self.region, "title": "Feature Store Operations",
                    },
                },
                # Row 6: Batch Transform Jobs (primary inference pattern)
                {
                    "type": "metric", "x": 0, "y": 30, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "TransformJobName", "*"],
                            [".", "MemoryUtilization", ".", "*"],
                        ],
                        "period": 300, "stat": "Average", "region": self.region, "title": "Batch Transform Resources",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 30, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "TransformTimeInSeconds", "TransformJobName", "*", {"stat": "Sum"}],
                            [".", "BillableTimeInSeconds", ".", "*", {"stat": "Sum"}],
                        ],
                        "period": 86400, "stat": "Sum", "region": self.region, "title": "Batch Transform Duration",
                    },
                },
                # Row 7: Forecast accuracy custom metrics (published by monitoring.py)
                {
                    "type": "metric", "x": 0, "y": 36, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["RetailForecasting/ModelQuality", "mae", "Segment", "overall"],
                            [".", "rmse", ".", "."],
                        ],
                        "period": 86400, "stat": "Average", "region": self.region,
                        "title": "Forecast Accuracy (MAE / RMSE)",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 36, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [["RetailForecasting/ModelQuality", "wape", "Segment", "overall"]],
                        "period": 86400, "stat": "Average", "region": self.region,
                        "title": "Forecast Accuracy (WAPE)",
                    },
                },
            ]
        }

        try:
            self.cloudwatch.put_dashboard(DashboardName=dashboard_name, DashboardBody=json.dumps(dashboard_body))
            logger.info(f"Comprehensive dashboard {dashboard_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")

        return dashboard_body

    def create_training_dashboard(self, dashboard_name: str = "training-jobs-monitoring"):
        """
        Create a dashboard specifically for monitoring training jobs.

        Args:
            dashboard_name: Name of the dashboard

        Returns:
            Dashboard configuration
        """
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "TrainingJobName", "*"],
                            [".", "MemoryUtilization", ".", "*"],
                            [".", "GPUUtilization", ".", "*"],
                        ],
                        "period": 300, "stat": "Average", "region": self.region, "title": "Training Job Resources",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "TrainingTimeInSeconds", "TrainingJobName", "*", {"stat": "Sum"}],
                            [".", "BillableTimeInSeconds", ".", "*", {"stat": "Sum"}],
                        ],
                        "period": 86400, "stat": "Sum", "region": self.region, "title": "Training Duration",
                    },
                },
            ]
        }

        try:
            self.cloudwatch.put_dashboard(DashboardName=dashboard_name, DashboardBody=json.dumps(dashboard_body))
            logger.info(f"Training dashboard {dashboard_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create training dashboard: {e}")

        return dashboard_body

    def create_endpoint_dashboard(self, dashboard_name: str = "endpoint-monitoring"):
        """
        Create a dashboard specifically for monitoring the (optional) real-time endpoint.

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
                            [".", "OverheadLatency", ".", "."],
                        ],
                        "period": 300, "stat": "Average", "region": self.region, "title": "Endpoint Latency",
                    },
                },
                {
                    "type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_dim, {"stat": "Sum"}],
                            [".", "4XXError", ".", ".", {"stat": "Sum"}],
                            [".", "5XXError", ".", ".", {"stat": "Sum"}],
                        ],
                        "period": 300, "stat": "Sum", "region": self.region, "title": "Endpoint Requests & Errors",
                    },
                },
                {
                    "type": "metric", "x": 0, "y": 6, "width": 24, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "EndpointName", endpoint_dim],
                            [".", "MemoryUtilization", ".", "."],
                        ],
                        "period": 300, "stat": "Average", "region": self.region,
                        "title": "Endpoint Resource Utilization",
                    },
                },
            ]
        }

        try:
            self.cloudwatch.put_dashboard(DashboardName=dashboard_name, DashboardBody=json.dumps(dashboard_body))
            logger.info(f"Endpoint dashboard {dashboard_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create endpoint dashboard: {e}")

        return dashboard_body

    def list_dashboards(self) -> list:
        """List all CloudWatch dashboards."""
        response = self.cloudwatch.list_dashboards()
        return [dashboard['DashboardName'] for dashboard in response['DashboardEntries']]

    def delete_dashboard(self, dashboard_name: str):
        """Delete a CloudWatch dashboard."""
        try:
            self.cloudwatch.delete_dashboards(DashboardNames=[dashboard_name])
            logger.info(f"Dashboard {dashboard_name} deleted")
        except Exception as e:
            logger.error(f"Failed to delete dashboard: {e}")


def main():
    """Main function to create all monitoring dashboards."""
    monitoring = InfrastructureMonitoring()

    monitoring.create_comprehensive_dashboard()
    monitoring.create_training_dashboard()
    monitoring.create_endpoint_dashboard()

    dashboards = monitoring.list_dashboards()
    print(f"\nCreated {len(dashboards)} dashboards:")
    for dashboard in dashboards:
        print(f"  - {dashboard}")


if __name__ == "__main__":
    main()
