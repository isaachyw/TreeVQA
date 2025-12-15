#!/usr/bin/env python3

import os
import subprocess
import json
from datetime import datetime
import argparse
import math

# Maximum number of tasks per batch
MAX_BATCH_SIZE = 5


# ------------------------------------------------------------------------------
# 1) LOAD MOLECULE CONFIGS FROM JSON
# ------------------------------------------------------------------------------
def load_configs_from_json(json_file_path):
    """
    Load the molecule configuration list from a JSON file.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_slurm_config(slurm_config_path):
    """
    Load SLURM configuration from a JSON file.
    """
    with open(slurm_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ------------------------------------------------------------------------------
# 2) SLURM SCRIPT TEMPLATE FOR CONCURRENT EXECUTION
# ------------------------------------------------------------------------------
def create_concurrent_slurm_script(configs, device, jobs_name, slurm, slurm_config):
    """
    Creates a single Slurm script that runs multiple configs concurrently.
    If slurm is False, only generates the execution section without SLURM headers.
    """
    num_configs = len(configs)
    # Number of concurrent tasks is limited by MAX_BATCH_SIZE
    concurrent_tasks = min(num_configs, MAX_BATCH_SIZE)

    # Create Slurm header (only if slurm is True)
    slurm_header = ""
    if slurm:
        # Get device-specific configuration
        device_config = slurm_config[device]
        total_cpus = device_config["total_cpus"]
        total_memory = device_config["total_memory"]

        # Distribute resources among concurrent tasks (limited by batch size)
        cpus_per_task = max(1, total_cpus // concurrent_tasks)

        total_batches = math.ceil(num_configs / MAX_BATCH_SIZE)

        # Build module load commands
        module_commands = "\n".join(
            [f"module load {module}" for module in device_config["modules"]]
        )

        # Common SBATCH directives
        common_header = f"""#!/bin/bash

#SBATCH --job-name={jobs_name}-concurrent-{num_configs}tasks
#SBATCH -A {slurm_config["account"]}
#SBATCH -C {device_config["constraint"]}
#SBATCH --qos {slurm_config["qos"]}
#SBATCH -t {slurm_config["time"]}
#SBATCH -N 1
#SBATCH --ntasks-per-node={concurrent_tasks}
#SBATCH --cpus-per-task={cpus_per_task}"""

        # Add GPU-specific directive if needed
        if device == "gpu":
            common_header += "\n#SBATCH --gpus-per-task=1"

        # Complete the header
        slurm_header = f"""{common_header}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user={slurm_config["email"]}
#SBATCH --output=slurm/%x-%j.out
#SBATCH --mem={total_memory}G

{module_commands}

echo "Loading modules done, starting {num_configs} tasks in {total_batches} batch(es) of max {MAX_BATCH_SIZE}..."
echo "Max {concurrent_tasks} concurrent tasks, each gets {cpus_per_task} CPUs"
echo "Starting at $(date)"
"""
    else:
        # No SLURM header, just a simple bash script header
        slurm_header = """#!/bin/bash

echo "Starting tasks at $(date)"
"""

    # Create concurrent Python commands
    python_commands = []
    for i, config in enumerate(configs):
        noisy_flag = "--noisy" if config.get("noisy", False) else ""
        cafqa_flag = "--cafqa" if config.get("cafqa", False) else ""
        truncate_ratio = config.get("truncate_ratio", 0.001)
        # Detect problem type (default to "molecule")
        problem_type = config.get("problem", "molecule")

        # Create the Python command based on problem type
        if problem_type == "ising":
            cmd = f"""python3 {config["run_file"]} -p ising \\
    --num_node {config["num_node"]} \\
    --magnetic_fields "{config["magnetic_fields"]}" \\
    --model_type {config["model_type"]} \\
    --vqe_iter {config["vqe_iter"]} \\
    --vqe_step_size {config["vqe_step"]} --repetition {config.get("repetition", 2)} \\
    --cap_window {config["window"]} --cap_warmup {config["warmup"]} \\
    --threshold {config["threshold"]} --result_dir {config.get("result_dir", "results")} \\
    --total_mode --check_interval 2 --ansatz {config.get("ansatz", "HEA")} --truncate_ratio {truncate_ratio} \\
    {noisy_flag} \\
    --gate_error {config.get("gate_error", 0.01)} \\
    {cafqa_flag} --cafqabudget {config.get("cafqabudget", 40)} \\
    --optimizer_method {config.get("optimizer_method", "SPSAP")}"""
        elif problem_type == "power":
            cmd = f"""python3 {config["run_file"]} -p power \\
    --num_vertices {config["num_vertices"]} \\
    --graph_slice "{config["graph_slice"]}" \\
    --weight_range "{config.get("weight_range")}" \\
    --vqe_iter {config["vqe_iter"]} \\
    --vqe_step_size {config["vqe_step"]} --repetition {config.get("repetition", 2)} \\
    --cap_window {config["window"]} --cap_warmup {config["warmup"]} \\
    --threshold {config["threshold"]} --result_dir {config.get("result_dir", "results")} \\
    --total_mode --check_interval 2 --ansatz {config.get("ansatz", "HEA")} --truncate_ratio {truncate_ratio} \\
    {noisy_flag} \\
    --gate_error {config.get("gate_error", 0.01)} \\
    {cafqa_flag} --cafqabudget {config.get("cafqabudget", 40)} \\
    --optimizer_method {config.get("optimizer_method", "SPSAP")}"""
        else:  # default to molecule
            cmd = f"""python3 {config["run_file"]} -p molecule \\
    --molecule_name {config["molecule"]} \\
    --molecule_slice "{config["molecule_slice"]}" \\
    --vqe_iter {config["vqe_iter"]} \\
    --vqe_step_size {config["vqe_step"]} --repetition {config.get("repetition", 2)} \\
    --cap_window {config["window"]} --cap_warmup {config["warmup"]} \\
    --threshold {config["threshold"]} --result_dir {config.get("result_dir", "results")} \\
    --total_mode --check_interval 2 --ansatz {config.get("ansatz", "HEA")} --truncate_ratio {truncate_ratio} \\
    {noisy_flag} \\
    --gate_error {config.get("gate_error", 0.01)} \\
    {cafqa_flag} --cafqabudget {config.get("cafqabudget", 40)} \\
    --optimizer_method {config.get("optimizer_method", "SPSAP")}"""

        cmd += " &"

        python_commands.append(cmd)

    # Create the concurrent execution section with batching
    total_batches = math.ceil(num_configs / MAX_BATCH_SIZE)
    execution_section = f"""
# Start tasks in batches of {MAX_BATCH_SIZE}
echo "Starting {num_configs} tasks in {total_batches} batch(es)..."
"""

    # Add commands in batches
    for batch_idx in range(total_batches):
        start_idx = batch_idx * MAX_BATCH_SIZE
        end_idx = min(start_idx + MAX_BATCH_SIZE, num_configs)
        batch_size = end_idx - start_idx

        execution_section += f"""
echo "=== Starting batch {batch_idx + 1}/{total_batches} ({batch_size} tasks) at $(date) ==="
"""

        # Add each Python command in this batch
        for i in range(start_idx, end_idx):
            # Generate task description based on problem type
            problem_type = configs[i].get("problem", "molecule")
            if problem_type == "ising":
                task_desc = f"{configs[i].get('num_node', 'ising')} nodes"
            elif problem_type == "power":
                task_desc = f"power-{configs[i].get('num_vertices', 'grid')}"
            else:
                task_desc = configs[i].get("molecule", "molecule")

            execution_section += f"""
echo "Starting task {i + 1}/{num_configs}: {task_desc} - {configs[i]["run_file"]}"
{python_commands[i]}
"""

        # Wait for this batch to complete before starting the next
        execution_section += f"""
echo "Waiting for batch {batch_idx + 1}/{total_batches} to complete..."
wait
echo "Batch {batch_idx + 1}/{total_batches} completed at $(date)"
"""

    # Add final completion message
    execution_section += """
echo "All batches completed at $(date)"
"""

    return slurm_header + execution_section


# ------------------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def submit_concurrent_job(configs, device, jobs_name, slurm, slurm_config):
    """
    1) Creates a single Slurm script for all configs to run concurrently.
    2) Writes it to a file.
    3) Submits it via `sbatch`.
    4) Returns (job_id, script_filename).
    """
    script_content = create_concurrent_slurm_script(
        configs, device, jobs_name, slurm, slurm_config
    )

    # Create script filename
    num_configs = len(configs)
    script_filename = f"concurrent_{jobs_name}_{num_configs}tasks_{device}.sh"

    # Write the Slurm script to a file
    with open(script_filename, "w") as f:
        f.write(script_content)

    # Make it executable
    os.chmod(script_filename, 0o755)

    print(f"Created concurrent job script: {script_filename}")
    print(f"Configure {num_configs} tasks concurrently on 1 {device} node")

    if slurm:
        # Submit the job
        result = subprocess.run(
            ["sbatch", script_filename], capture_output=True, text=True
        )
        if result.stderr:
            print("STDERR:", result.stderr)

        sbatch_output = result.stdout.strip()
        print("SBATCH OUTPUT:", sbatch_output)

        # Parse job ID
        job_id = None
        if "Submitted batch job" in sbatch_output:
            job_id = sbatch_output.split()[-1]

        # remove the script file
        os.remove(script_filename)
        return job_id, script_filename
    else:
        # not remove the script
        return None, script_filename


def log_concurrent_submission(logfile, configs, job_id, device):
    """
    Logs the concurrent job submission details.
    """
    with open(logfile, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"[{timestamp}] Concurrent JobID={job_id}, Device={device}, NumTasks={len(configs)}\n"
        )
        for i, config in enumerate(configs):
            # Generate task description based on problem type
            problem_type = config.get("problem", "molecule")
            if problem_type == "ising":
                task_desc = f"{config.get('num_node', 'ising')} nodes"
            elif problem_type == "power":
                task_desc = f"power-{config.get('num_vertices', 'grid')}"
            else:
                task_desc = config.get("molecule", "molecule")
            f.write(f"  Task {i + 1}: {task_desc} - {config['run_file']}\n")
        f.write("\n")


# ------------------------------------------------------------------------------
# 4) MAIN FUNCTION
# ------------------------------------------------------------------------------
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Create shell script from configuration file. If --slurm is set, submit the script as concurrent tasks in a single SLURM job."
    )
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        help="Path to the configuration file (default: molecule_configs.json)",
    )
    parser.add_argument(
        "device",
        type=str,
        nargs="?",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",  # by default is False
        help="Submit the job to SLURM",
    )
    parser.add_argument(
        "--slurm-config",
        type=str,
        default="slurm_config.json",
        help="Path to SLURM configuration file (default: slurm_config.json)",
    )
    # Parse arguments
    args = parser.parse_args()
    slurm_config = None
    # if slurm is True, then we need to load the slurm config file
    if args.slurm:
        # Load SLURM config file
        slurm_config_file = args.slurm_config
        if not os.path.exists(slurm_config_file):
            print(f"Error: SLURM config file {slurm_config_file} not found!")
            return
        slurm_config = load_slurm_config(slurm_config_file)
    # Create necessary directories
    os.makedirs("slurm", exist_ok=True)

    # Load configs from JSON file
    json_config_file = args.config_file
    if not os.path.exists(json_config_file):
        print(f"Error: Config file {json_config_file} not found!")
        return

    global_config = load_configs_from_json(json_config_file)
    jobs_name = global_config["job-name"]
    molecule_configs = global_config["configs"]

    # Validate we have configs
    if not molecule_configs:
        print("Error: No configurations found!")
        return

    num_batches = math.ceil(len(molecule_configs) / MAX_BATCH_SIZE)
    print(
        f"Found {len(molecule_configs)} configurations to run in {num_batches} batch(es) (max {MAX_BATCH_SIZE} per batch)"
    )

    # Submit the concurrent job
    job_id, script_filename = submit_concurrent_job(
        molecule_configs, args.device, jobs_name, args.slurm, slurm_config
    )
    if args.slurm:
        if job_id:
            print(f"Successfully submitted concurrent job with JobID={job_id}")
            print(f"Script file: {script_filename}")

            # Log the submission
            log_file = f"slurm/{jobs_name}_concurrent_submissions.log"
            log_concurrent_submission(log_file, molecule_configs, job_id, args.device)

            print(f"\nMonitor job status with:")
            print(f"  squeue -j {job_id}")
            print(f"  scontrol show job {job_id}")
            print(f"\nView output with:")
            print(
                f"  tail -f slurm/{jobs_name}-concurrent-{len(molecule_configs)}tasks-{job_id}.out"
            )

        else:
            print("Failed to parse job ID from submission.")
    else:
        print(f"Command to run save at {script_filename}")
    return 0


if __name__ == "__main__":
    main()
