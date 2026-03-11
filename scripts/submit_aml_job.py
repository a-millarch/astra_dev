"""
Submit an ASTRA training job to Azure ML (SDK v2).

Usage:
    # Full pipeline (pretrain + finetune + eval + multicurve + comprehensive-eval):
    python scripts/submit_aml_job.py --pretrain --multicurve --comprehensive-eval

    # Dry run (preview config without submitting):
    python scripts/submit_aml_job.py --pretrain --multicurve --comprehensive-eval --dry-run

    # Finetune + eval only:
    python scripts/submit_aml_job.py

    # Custom compute target:
    python scripts/submit_aml_job.py --pretrain --compute my-cluster
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONDA_FILE = PROJECT_ROOT / "environment_gpu.yml"
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"


def get_ml_client() -> MLClient:
    """Create MLClient from env vars or config.json."""
    credential = DefaultAzureCredential()

    sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    rg = os.environ.get("AZURE_RESOURCE_GROUP")
    ws = os.environ.get("AZURE_WORKSPACE_NAME")

    if all([sub_id, rg, ws]):
        return MLClient(credential, sub_id, rg, ws)

    return MLClient.from_config(credential=credential)


def build_training_command(args) -> str:
    """Build the training CLI command from flags."""
    parts = ["python -m astra.training.train"]

    if args.pretrain:
        parts.append("--pretrain")
    if args.finetune:
        parts.append("--finetune")
    else:
        parts.append("--no-finetune")
    if args.eval:
        parts.append("--eval")
    else:
        parts.append("--no-eval")
    if args.multicurve:
        parts.append("--multicurve")
    if args.comprehensive_eval:
        parts.append("--comprehensive-eval")

    return " ".join(parts)


def build_display_name(args) -> str:
    """Generate DDMMYYYY_description display name."""
    date_prefix = datetime.now().strftime("%d%m%Y")
    if args.description:
        return f"{date_prefix}_{args.description}"

    stages = []
    if args.pretrain:
        stages.append("pt")
    if args.finetune:
        stages.append("ft")
    if args.eval:
        stages.append("eval")
    if args.multicurve:
        stages.append("mc")
    if args.comprehensive_eval:
        stages.append("compeval")
    suffix = "_".join(stages) if stages else "run"
    return f"{date_prefix}_{suffix}"


def validate_compute(ml_client: MLClient, compute_name: str):
    """Check that the compute target exists, listing alternatives on failure."""
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"Compute '{compute_name}': state={compute.provisioning_state}, "
              f"size={getattr(compute, 'size', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Compute '{compute_name}' not found: {e}")
        print("Available computes:")
        for c in ml_client.compute.list():
            print(f"  - {c.name} ({c.type}, {getattr(c, 'size', 'N/A')})")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit ASTRA training job to Azure ML"
    )

    # Training flags (mirror astra.training.train)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multicurve", action="store_true", default=False)
    parser.add_argument("--comprehensive-eval", action=argparse.BooleanOptionalAction, default=False)

    # Azure ML options
    parser.add_argument("--compute", type=str, default="gpu-t4-cluster",
                        help="AzureML compute target name")
    parser.add_argument("--experiment", type=str, default="hpt",
                        help="Experiment name (default: hpt)")
    parser.add_argument("--description", type=str, default=None,
                        help="Short suffix for display name (default: auto from flags)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print job config without submitting")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Build job config (before connecting, for dry-run support) ---
    training_cmd = build_training_command(args)
    display_name = build_display_name(args)

    pipeline_stages = []
    if args.pretrain:
        pipeline_stages.append("pretrain")
    if args.finetune:
        pipeline_stages.append("finetune")
    if args.eval:
        pipeline_stages.append("eval")
    if args.multicurve:
        pipeline_stages.append("multicurve")
    if args.comprehensive_eval:
        pipeline_stages.append("comprehensive-eval")

    tags = {
        "CSTAR": "",
        "pipeline_stages": "+".join(pipeline_stages),
    }

    env = Environment(
        name="astra-gpu",
        description="ASTRA GPU training environment (PyTorch 2.3.1, CUDA 12.1, T4)",
        conda_file=str(CONDA_FILE),
        image=BASE_IMAGE,
    )

    job = command(
        code=str(PROJECT_ROOT),
        command=training_cmd,
        environment=env,
        compute=args.compute,
        display_name=display_name,
        experiment_name=args.experiment,
        tags=tags,
        description=f"ASTRA training: {training_cmd}",
    )

    if args.dry_run:
        print("\n=== DRY RUN (job not submitted) ===")
        print(f"  Display name:  {display_name}")
        print(f"  Experiment:    {args.experiment}")
        print(f"  Compute:       {args.compute}")
        print(f"  Command:       {training_cmd}")
        print(f"  Environment:   {env.name} (image: {BASE_IMAGE})")
        print(f"  Conda file:    {CONDA_FILE}")
        print(f"  Code path:     {PROJECT_ROOT}")
        print(f"  Tags:          {tags}")
        return

    # --- Connect and submit ---
    print("Connecting to Azure ML workspace...")
    try:
        ml_client = get_ml_client()
    except Exception as e:
        print(f"ERROR: Failed to connect to Azure ML: {e}")
        print("Ensure you are logged in (az login) or on an Azure ML compute instance.")
        sys.exit(1)

    print(f"Connected to workspace: {ml_client.workspace_name}")
    validate_compute(ml_client, args.compute)

    print(f"\nSubmitting job '{display_name}'...")
    print(f"  Command: {training_cmd}")
    returned_job = ml_client.jobs.create_or_update(job)

    print(f"\nJob submitted successfully!")
    print(f"  Name:       {returned_job.name}")
    print(f"  Display:    {returned_job.display_name}")
    print(f"  Status:     {returned_job.status}")
    print(f"  Studio URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
