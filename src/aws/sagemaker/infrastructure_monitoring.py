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

    @staticmethod
    def _search(expression: str, stat: str, period: int, label: str, metric_id: str) -> list:
        """A CloudWatch SEARCH() metric row: one line per matching job/segment."""
        return [{"expression": f"SEARCH('{expression}', '{stat}', {period})", "label": label, "id": metric_id}]

    def _bucket_name(self) -> str:
        account_id = self.account_id or boto3.client('sts', region_name=self.region).get_caller_identity()['Account']
        return f"sagemaker-{self.region}-{account_id}"

    def build_comprehensive_dashboard(self, bucket_name: str) -> dict:
        """
        Dashboard body for the ML system as it actually runs: SageMaker
        Pipeline executions, the training / processing / batch-transform jobs
        they launch (instance metrics are published per job host under the
        /aws/sagemaker/*Jobs namespaces), Feature Store, S3, forecast accuracy
        published by monitoring_reports.py, and the optional endpoint.
        SEARCH() expressions pick up every job automatically, so new runs
        appear without editing the dashboard.
        """
        r = self.region
        pipeline_dims = '{AWS/Sagemaker/ModelBuildingPipeline,PipelineName}'
        step_dims = '{AWS/Sagemaker/ModelBuildingPipeline,PipelineName,StepName}'
        quality = '{RetailForecasting/ModelQuality,Segment}'

        def widget(x, y, title, metrics, width=12, height=6, **props):
            return {"type": "metric", "x": x, "y": y, "width": width, "height": height,
                    "properties": {"metrics": metrics, "region": r, "title": title, **props}}

        def job_widget(x, y, title, namespace):
            return widget(x, y, title,
                          self._search(f'{{{namespace},Host}} MetricName="CPUUtilization"', 'Average', 60,
                                       'CPU %', 'cpu') +
                          self._search(f'{{{namespace},Host}} MetricName="MemoryUtilization"', 'Average', 60,
                                       'Memory %', 'mem'),
                          view="timeSeries", period=60)

        widgets = [
            {"type": "text", "x": 0, "y": 0, "width": 24, "height": 2, "properties": {"markdown": (
                "# Retail demand forecasting: ML system health\n"
                "Pipeline `retail-forecasting-pipeline` (Preprocess > Train > Evaluate > WAPE gate > Register) | "
                "batch transform inference | Feature Store `retail-inventory-features-*` | "
                f"S3 `{bucket_name}/{self.base_prefix}`")}},
            # Row 1: CI/CD DAG health
            widget(0, 2, "Pipeline executions (succeeded / failed)",
                   self._search(f'{pipeline_dims} MetricName="ExecutionSucceeded"', 'Sum', 3600, 'Succeeded', 'ok') +
                   self._search(f'{pipeline_dims} MetricName="ExecutionFailed"', 'Sum', 3600, 'Failed', 'fail'),
                   view="bar", period=3600),
            widget(12, 2, "Pipeline step failures by step",
                   self._search(f'{step_dims} MetricName="StepFailed"', 'Sum', 3600, '', 'stepfail'),
                   view="bar", period=3600),
            # Row 2: training and processing jobs
            job_widget(0, 8, "Training jobs: CPU / memory per job", '/aws/sagemaker/TrainingJobs'),
            job_widget(12, 8, "Processing jobs (preprocess, evaluate, Model Monitor): CPU / memory",
                       '/aws/sagemaker/ProcessingJobs'),
            # Row 3: inference and features
            job_widget(0, 14, "Batch transform jobs: CPU / memory", '/aws/sagemaker/TransformJobs'),
            widget(12, 14, "Feature Store operations",
                   self._search('{AWS/SageMaker/FeatureStore,FeatureGroupName,OperationName} MetricName="Invocations"',
                                'Sum', 300, '', 'fsinv'),
                   view="timeSeries", period=300),
            # Row 4: model quality from monitoring_reports.py
            widget(0, 20, "Forecast accuracy: model vs seasonal-naive (MAE / RMSE)", [
                ["RetailForecasting/ModelQuality", "mae", "Segment", "overall", {"label": "Model MAE"}],
                [".", "rmse", ".", ".", {"label": "Model RMSE"}],
                [".", "mae", ".", "seasonal_naive", {"label": "Seasonal-naive MAE"}],
                [".", "rmse", ".", "seasonal_naive", {"label": "Seasonal-naive RMSE"}],
            ], view="bar", stat="Average", period=86400),
            widget(12, 20, "WAPE by segment (overall, baseline, per store)",
                   self._search(f'{quality} MetricName="wape"', 'Average', 86400, '', 'wape'),
                   view="bar", period=86400),
            # Row 5: storage and optional endpoint
            widget(0, 26, "S3 project bucket", [
                ["AWS/S3", "BucketSizeBytes", "BucketName", bucket_name, "StorageType", "StandardStorage"],
                [".", "NumberOfObjects", ".", ".", "StorageType", "AllStorageTypes", {"yAxis": "right"}],
            ], stat="Average", period=86400),
            widget(12, 26, "Real-time endpoint (only when deployed): latency / errors",
                   self._search('{AWS/SageMaker,EndpointName,VariantName} MetricName="ModelLatency"',
                                'Average', 300, 'Latency', 'lat') +
                   self._search('{AWS/SageMaker,EndpointName,VariantName} MetricName="Invocation5XXErrors"',
                                'Sum', 300, '5XX', 'err'),
                   view="timeSeries", period=300),
            # Row 6: metrics printed by the training script
            {"type": "log", "x": 0, "y": 32, "width": 24, "height": 6, "properties": {
                "region": r, "title": "Training job metrics (validation / test / seasonal-naive)", "view": "table",
                "query": ("SOURCE '/aws/sagemaker/TrainingJobs' | fields @timestamp, @logStream, @message "
                          "| filter @message like /Validation metrics|Test metrics/ "
                          "| sort @timestamp desc | limit 20"),
            }},
        ]
        return {"widgets": widgets}

    def create_comprehensive_dashboard(self, dashboard_name: str = "ml-system-monitoring"):
        """
        Create the comprehensive CloudWatch dashboard for the ML system.

        Args:
            dashboard_name: Name of the dashboard

        Returns:
            Dashboard configuration
        """
        dashboard_body = self.build_comprehensive_dashboard(self._bucket_name())
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
        namespace = '{/aws/sagemaker/TrainingJobs,Host}'
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": (self._search(f'{namespace} MetricName="CPUUtilization"', 'Average', 60, 'CPU %', 'cpu')
                                    + self._search(f'{namespace} MetricName="MemoryUtilization"', 'Average', 60,
                                                   'Memory %', 'mem')
                                    + self._search(f'{namespace} MetricName="DiskUtilization"', 'Average', 60,
                                                   'Disk %', 'disk')),
                        "period": 60, "region": self.region, "title": "Training Job Resources",
                    },
                },
                {
                    "type": "log", "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "region": self.region, "title": "Training job metrics", "view": "table",
                        "query": ("SOURCE '/aws/sagemaker/TrainingJobs' | fields @timestamp, @logStream, @message "
                                  "| filter @message like /Validation metrics|Test metrics/ "
                                  "| sort @timestamp desc | limit 20"),
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
