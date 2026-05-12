import multiprocessing as mp
import shlex
from typing import Callable, Optional

from scripts.run import main as run_main
from scripts.run import setup_arguments as run_setup_arguments


def _worker(job_str: str, setup_arguments: Callable, main: Callable):
    job_args_list = shlex.split(job_str)
    assert job_args_list[0] == 'JOB'
    del job_args_list[0]  # Delete the dummy "JOB" prefix
    print(job_args_list)
    job_args = setup_arguments(args=job_args_list)

    main(job_args)


def main_njobs(
    job_specs,
    njobs: int,
    start_method: str = "spawn",
    setup_arguments: Optional[Callable] = None,
    main: Optional[Callable] = None,
):
    try:
        mp.set_start_method(start_method, force=True)
    except RuntimeError:
        pass

    setup_arguments = setup_arguments or run_setup_arguments
    main = main or run_main

    with mp.Pool(processes=njobs) as pool:
        pool.starmap(_worker, [(spec, setup_arguments, main) for spec in job_specs])
