"""
SageMaker Pipeline (the project's CI/CD DAG) for the retail demand forecaster.

    Preprocess -> TrainModel -> EvaluateModel -> CheckQualityGate
                                                  |-- pass --> RegisterModel (Model Registry)
                                                  '-- fail --> QualityGateFailed (Fail step)

The quality gate reads evaluation.json (written by scripts/train.py from the
held-out test split) and requires both:
  - test WAPE <= MaxTestWAPE
  - relative WAPE improvement over the seasonal-naive baseline >= MinWAPEImprovement

Usage (from SageMaker Studio, which supplies the execution role):
    python -m src.aws.sagemaker.pipeline upsert
    python -m src.aws.sagemaker.pipeline run --wait                  # passes the gate, registers
    python -m src.aws.sagemaker.pipeline run --max-wape 0.5 --wait   # fails the gate (demo)
    python -m src.aws.sagemaker.pipeline status
"""

import argparse
import json
import logging
import os
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PIPELINE_NAME = 'retail-forecasting-pipeline'
MODEL_PACKAGE_GROUP = 'retail-inventory-forecasting-models'
BASE_PREFIX = 'aai540/retail-forecasting'
SKLEARN_VERSION = '1.2-1'

DEFAULT_MAX_TEST_WAPE = 0.70
DEFAULT_MIN_WAPE_IMPROVEMENT = 0.10

SAGEMAKER_DIR = Path(__file__).parent
TRAIN_SCRIPTS_DIR = SAGEMAKER_DIR / 'scripts'
PIPELINE_SCRIPTS_DIR = SAGEMAKER_DIR / 'pipeline_scripts'


def resolve_role(sagemaker_session) -> str:
    role = os.getenv('AWS_ROLE_ARN')
    if role:
        return role
    return sagemaker.get_execution_role(sagemaker_session=sagemaker_session)


def resolve_bucket(region: str) -> str:
    account_id = os.getenv('AWS_ACCOUNT_ID') or boto3.client('sts', region_name=region).get_caller_identity()['Account']
    return f'sagemaker-{region}-{account_id}'


def build_pipeline(role: str, bucket: str, region: str, pipeline_session: PipelineSession = None) -> Pipeline:
    """Define (but do not create) the SageMaker Pipeline."""
    if pipeline_session is None:
        pipeline_session = PipelineSession(boto_session=boto3.session.Session(region_name=region),
                                           default_bucket=bucket)
    s3_root = f's3://{bucket}/{BASE_PREFIX}'

    input_data = ParameterString('InputDataUrl', default_value=f'{s3_root}/raw/retail_store_inventory.csv')
    processing_instance = ParameterString('ProcessingInstanceType', default_value='ml.m5.xlarge')
    training_instance = ParameterString('TrainingInstanceType', default_value='ml.m5.xlarge')
    ridge_alpha = ParameterFloat('RidgeAlpha', default_value=1.0)
    max_test_wape = ParameterFloat('MaxTestWAPE', default_value=DEFAULT_MAX_TEST_WAPE)
    min_improvement = ParameterFloat('MinWAPEImprovement', default_value=DEFAULT_MIN_WAPE_IMPROVEMENT)
    approval_status = ParameterString('ModelApprovalStatus', default_value='PendingManualApproval')

    def processor(base_job_name):
        return FrameworkProcessor(
            estimator_cls=SKLearn, framework_version=SKLEARN_VERSION, py_version='py3',
            role=role, instance_type=processing_instance, instance_count=1,
            base_job_name=base_job_name, sagemaker_session=pipeline_session,
        )

    # 1. Preprocess: clean + chronological split
    step_preprocess = ProcessingStep(
        name='Preprocess',
        step_args=processor('retail-forecasting-preprocess').run(
            code='preprocess.py', source_dir=str(PIPELINE_SCRIPTS_DIR),
            inputs=[ProcessingInput(source=input_data, destination='/opt/ml/processing/input')],
            outputs=[
                ProcessingOutput(output_name=name, source=f'/opt/ml/processing/{name}',
                                 destination=f'{s3_root}/pipeline/splits/{name}')
                for name in ('train', 'validation', 'test')
            ],
        ),
    )
    split_uri = {
        name: step_preprocess.properties.ProcessingOutputConfig.Outputs[name].S3Output.S3Uri
        for name in ('train', 'validation', 'test')
    }

    # 2. Train (Ridge, the batch-transform model) and score the test split
    estimator = TensorFlow(
        entry_point='train.py', source_dir=str(TRAIN_SCRIPTS_DIR),
        role=role, instance_type=training_instance, instance_count=1,
        framework_version='2.13', py_version='py310',
        hyperparameters={'model_type': 'ridge', 'ridge_alpha': ridge_alpha},
        output_path=f'{s3_root}/models/', base_job_name='retail-forecasting-pipeline-train',
        sagemaker_session=pipeline_session,
        # Without this, the TensorFlow estimator auto-injects its own 'model_dir'
        # hyperparameter (an S3 URI, meant for native TF distributed checkpointing).
        # train.py's own --model_dir argparse argument then silently receives that
        # S3 URI instead of the real SM_MODEL_DIR (/opt/ml/model), so model.joblib
        # gets written to a bogus local directory mimicking the S3 path and never
        # reaches /opt/ml/model - SageMaker then has nothing to package into
        # model.tar.gz, and EvaluateModel fails downstream with "No S3 objects
        # found". model_dir=False disables the auto-injection so train.py's own
        # SM_MODEL_DIR-derived default is used instead.
        model_dir=False,
    )
    step_train = TrainingStep(
        name='TrainModel',
        step_args=estimator.fit({
            name: TrainingInput(s3_data=uri, content_type='application/x-parquet')
            for name, uri in split_uri.items()
        }),
    )
    model_artifact = step_train.properties.ModelArtifacts.S3ModelArtifacts

    # 3. Evaluate: surface evaluation.json from the artifact as a property file
    evaluation_report = PropertyFile(name='EvaluationReport', output_name='evaluation', path='evaluation.json')
    step_evaluate = ProcessingStep(
        name='EvaluateModel',
        step_args=processor('retail-forecasting-evaluate').run(
            code='evaluate.py', source_dir=str(PIPELINE_SCRIPTS_DIR),
            inputs=[ProcessingInput(source=model_artifact, destination='/opt/ml/processing/model')],
            outputs=[ProcessingOutput(output_name='evaluation', source='/opt/ml/processing/evaluation',
                                      destination=f'{s3_root}/pipeline/evaluation')],
        ),
        property_files=[evaluation_report],
    )
    evaluation_uri = step_evaluate.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri
    test_wape = JsonGet(step_name=step_evaluate.name, property_file=evaluation_report, json_path='test.wape')
    improvement = JsonGet(step_name=step_evaluate.name, property_file=evaluation_report,
                          json_path='wape_improvement_vs_baseline')

    # 4a. Register the passing model (with its batch-transform inference code)
    model = SKLearnModel(
        model_data=model_artifact, role=role,
        entry_point='batch_transform.py', source_dir=str(TRAIN_SCRIPTS_DIR),
        framework_version=SKLEARN_VERSION, py_version='py3', sagemaker_session=pipeline_session,
    )
    step_register = ModelStep(
        name='RegisterModel',
        step_args=model.register(
            content_types=['text/csv'], response_types=['text/csv'],
            inference_instances=['ml.m5.large', 'ml.m5.xlarge'], transform_instances=['ml.m5.xlarge'],
            model_package_group_name=MODEL_PACKAGE_GROUP, approval_status=approval_status,
            model_metrics=ModelMetrics(model_statistics=MetricsSource(
                s3_uri=Join(on='/', values=[evaluation_uri, 'evaluation.json']),
                content_type='application/json',
            )),
            description='Ridge demand forecaster registered by retail-forecasting-pipeline after passing the WAPE gate',
        ),
    )

    # 4b. Stop the run with an explicit failure when the gate is not met
    step_fail = FailStep(
        name='QualityGateFailed',
        error_message=Join(on=' ', values=[
            'Quality gate failed: test WAPE', test_wape, 'must be <=', max_test_wape,
            'and WAPE improvement vs seasonal-naive', improvement, 'must be >=', min_improvement,
        ]),
    )

    step_gate = ConditionStep(
        name='CheckQualityGate',
        conditions=[
            ConditionLessThanOrEqualTo(left=test_wape, right=max_test_wape),
            ConditionGreaterThanOrEqualTo(left=improvement, right=min_improvement),
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data, processing_instance, training_instance, ridge_alpha,
                    max_test_wape, min_improvement, approval_status],
        steps=[step_preprocess, step_train, step_evaluate, step_gate],
        sagemaker_session=pipeline_session,
    )


def ensure_raw_data(bucket: str, region: str, local_csv: str = 'data/raw/retail_store_inventory.csv'):
    """Upload the raw CSV to the pipeline's default input location if it is missing."""
    s3 = boto3.client('s3', region_name=region)
    key = f'{BASE_PREFIX}/raw/retail_store_inventory.csv'
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError:
        logger.info(f'Uploading {local_csv} to s3://{bucket}/{key}')
        s3.upload_file(local_csv, bucket, key)


def print_steps(steps):
    for step in steps:
        reason = f" ({step['FailureReason']})" if step.get('FailureReason') else ''
        print(f"    {step['StepName']}: {step['StepStatus']}{reason}")


def print_recent_executions(sm, max_results: int = 5):
    summaries = sm.list_pipeline_executions(PipelineName=PIPELINE_NAME, MaxResults=max_results)
    for summary in summaries['PipelineExecutionSummaries']:
        arn = summary['PipelineExecutionArn']
        print(f"{summary.get('PipelineExecutionDisplayName', arn.split('/')[-1])}: {summary['PipelineExecutionStatus']}")
        print_steps(sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)['PipelineExecutionSteps'])


def wait_for_execution(execution) -> str:
    """Block until the execution finishes; print step statuses and return the final status."""
    try:
        execution.wait(delay=30, max_attempts=120)
    except Exception as e:  # WaiterError when the execution ends in Failed
        logger.warning(f'Execution did not succeed: {e}')
    print_steps(execution.list_steps())
    status = execution.describe()['PipelineExecutionStatus']
    print(f'Execution status: {status}')
    return status


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('upsert', help='Create or update the pipeline definition')
    run = sub.add_parser('run', help='Upsert and start an execution')
    run.add_argument('--max-wape', type=float, help=f'Gate: maximum test WAPE (default {DEFAULT_MAX_TEST_WAPE})')
    run.add_argument('--min-improvement', type=float,
                     help=f'Gate: minimum WAPE improvement vs seasonal-naive (default {DEFAULT_MIN_WAPE_IMPROVEMENT})')
    run.add_argument('--display-name', help='Execution display name shown in Studio')
    run.add_argument('--wait', action='store_true', help='Block until the execution finishes')
    sub.add_parser('status', help='Show recent executions and their step statuses')
    sub.add_parser('definition', help='Print the pipeline definition JSON')
    args = parser.parse_args()

    region = os.getenv('AWS_REGION', 'us-east-1')
    bucket = resolve_bucket(region)
    session = PipelineSession(boto_session=boto3.session.Session(region_name=region), default_bucket=bucket)
    sm = boto3.client('sagemaker', region_name=region)

    if args.command == 'status':
        print_recent_executions(sm)
        return

    pipeline = build_pipeline(resolve_role(session), bucket, region, session)
    if args.command == 'definition':
        print(json.dumps(json.loads(pipeline.definition()), indent=2))
        return

    ensure_raw_data(bucket, region)
    pipeline.upsert(role_arn=resolve_role(session), description='Retail demand forecasting CI/CD DAG with WAPE quality gate')
    logger.info(f'Pipeline {PIPELINE_NAME} upserted')
    if args.command == 'upsert':
        return

    overrides = {}
    if args.max_wape is not None:
        overrides['MaxTestWAPE'] = args.max_wape
    if args.min_improvement is not None:
        overrides['MinWAPEImprovement'] = args.min_improvement
    execution = pipeline.start(parameters=overrides or None, execution_display_name=args.display_name)
    print(f'Started execution: {execution.arn}')
    if args.wait and wait_for_execution(execution) != 'Succeeded':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
