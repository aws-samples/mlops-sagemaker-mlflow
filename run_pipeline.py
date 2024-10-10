# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A CLI to create or update and run pipelines."""
from __future__ import absolute_import

import argparse
import json
import sys
import traceback
import time
import ast

from pipelines._utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags
import boto3


def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    parser.add_argument(
        "-pipeline-name",
        "--pipeline-name",
        dest="pipeline_name",
        default=None,
        help="""name of the pipeline to run""",
    )
    args = parser.parse_args()

    tags = convert_struct(args.tags)
    kwarg_dict = convert_struct(args.kwargs)
    
    pipelineParameters = []
    for key, value in kwarg_dict.items():
        item = { "Name" : key, "Value" : value }
        pipelineParameters.append(item)
            
    try:
        sagemaker_client = boto3.client("sagemaker")
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=args.pipeline_name,
            PipelineParameters=pipelineParameters
        )
        pipeline_execution_arn = response["PipelineExecutionArn"]
        status = None
        while True:
            response = sagemaker_client.describe_pipeline_execution(PipelineExecutionArn=pipeline_execution_arn)
            status = response["PipelineExecutionStatus"]
            if status in ['Stopped', 'Failed', 'Succeeded']:
                break
            time.sleep(10)
        print(f"pipeline execution status: {status}")
        print("Waiting for the execution to finish...")
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
