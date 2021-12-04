#!/usr/bin/python3

import argparse
import os
import stat
import subprocess
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class SlurmJob:
    def __init__(
        self,
        name,
        time,
        gpu,
        num_gpus,
        memory,
        email,
        on_slurm,
        index=0,
    ):
        self.name = f"{name}-{index}"
        self.email = email
        self.time = time

        days, hours, minutes = list(
            map(int, [time.split("-")[0]] + time.split("-")[1].split(":"))
        )
        if "2080" in gpu:
            self.gpu = "gpu-2080ti-dev" if hours < 12 else "gpu-2080ti"
        else:
            self.gpu = "gpu-v100"
        self.num_gpus = num_gpus
        self.memory = memory
        self.on_slurm = on_slurm

    @property
    def resource_config_string(self):

        if not Path("logs").exists():
            os.mkdir("logs")
        config_string = """
#SBATCH --job-name={name}                   # Name of the job
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=2                   # Number of CPU cores per task
#SBATCH --nodes=1                           # Ensure that all cores are on one machine
#SBATCH --time={time}                       # Runtime in D-HH:MM
#SBATCH --partition={gpu}                   # Partition to submit to
#SBATCH --gres=gpu:{num_gpus}               # Number of requested GPUs
#SBATCH --mem-per-cpu={memory}              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/{name}.%j.out              # File to which STDOUT will be written
#SBATCH --error=logs/{name}.%j.err               # File to which STDERR will be written
#SBATCH --mail-type=ALL                     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={email}                 # Email to which notifications will be sent
        """.format(
            name=self.name,
            time=self.time,
            gpu=self.gpu,
            num_gpus=self.num_gpus,
            memory=self.memory,
            email=self.email,
        )
        return config_string

    @property
    def singularity_run_command(self):
        cmd_string = """
singularity run \
--nv \
--env-file .env \
--no-home  \
--bind $SCRATCH:/data/,/home/sinz/_shared/:/sinz_shared,$HOME/projects/:$HOME/projects/  \
singularity_img.sif  \
./run.py
        """
        return cmd_string

    @property
    def slurm_command(self):
        cmd_string = """
scontrol show job $SLURM_JOB_ID  # print some info
        """
        return cmd_string

    def run(self):

        if self.on_slurm:
            slurm_job_bash_file = f"./{self.name}.sh"
            slurm_job_bash_file_content = (
                "#!/bin/bash \n \n"
                + self.resource_config_string
                + "\n"
                + self.slurm_command
                + "\n"
                + self.singularity_run_command
            )

            with open(slurm_job_bash_file, "w") as f:
                f.write(slurm_job_bash_file_content)

            os.chmod(slurm_job_bash_file, stat.S_IRWXU)

            try:
                print(
                    subprocess.check_output("sbatch " + slurm_job_bash_file, shell=True)
                )
            finally:
                # remove the bash file
                os.remove(slurm_job_bash_file)

        else:
            print(subprocess.check_output(self.singularity_run_command, shell=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running jobs on SLURM cluster")
    parser.add_argument(
        "--njobs",
        dest="num_jobs",
        action="store",
        default=1,
        type=int,
        help="",
    )
    parser.add_argument(
        "--name",
        dest="name",
        action="store",
        default="noname",
        type=str,
        help="",
    )
    parser.add_argument(
        "--time",
        dest="time",
        action="store",
        default="0-00:00",
        type=str,
        help="time to complete each job. Specify in the following format: D-HH:MM",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store",
        default="gpu-2080ti",
        type=str,
        help="",
    )
    parser.add_argument(
        "--ngpus",
        dest="num_gpus",
        action="store",
        default=1,
        type=int,
        help="",
    )
    parser.add_argument(
        "--memory",
        dest="memory",
        action="store",
        default=3000,
        type=int,
        help="",
    )
    parser.add_argument(
        "--force-rebuild",
        dest="force_rebuild",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--email",
        dest="email",
        action="store",
        default=os.getenv("EMAIL"),
        type=str,
        help="",
    )
    parser.add_argument(
        "--slurm",
        dest="slurm",
        default=True,
        type=str2bool,
        help="Specify whether this is a job on SLURM or a normal GPU server.",
    )

    args = parser.parse_args()
    if not Path("./singularity_img.sif").exists() or args.force_rebuild:
        subprocess.check_output(
            "singularity build --force --fakeroot singularity_img.sif ./singularity.def",
            shell=True,
        )

    for job_index in range(args.num_jobs):
        job = SlurmJob(
            args.name,
            args.time,
            args.gpu,
            args.num_gpus,
            args.memory,
            args.email,
            args.slurm,
            index=job_index,
        )
        job.run()
