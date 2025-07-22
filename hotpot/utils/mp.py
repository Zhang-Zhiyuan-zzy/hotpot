"""
python v3.9.0
@Project: hotpot
@File   : mp
@Auther : Zhiyuan Zhang
@Data   : 2024/12/27
@Time   : 18:59
"""
import time
import logging
from tqdm import tqdm
from typing import Iterable, Callable, Optional, Any
import multiprocessing as mp

from . import fmt_print


def decorator(func: Callable) -> Callable:
    def _target(queue, *a, **kw_):
        r_ = func(*a, **kw_)
        logging.debug(r_)
        # time.sleep(30)
        queue.put(r_)

    return _target


def mp_run(
        func: Callable,
        args: Iterable[tuple],
        kwargs: Iterable[dict] = None,
        nproc: int = None,
        desc: str = '',
        timeout: Optional[float] = None,
        error_to_None: bool = True,
        lazy_iter: bool = False,
        early_exit_condition: Optional[Callable[[dict, int], bool]] = None,
        branch_jobs: Iterable[Callable[[dict, dict, int], None]] = None,
):

    if nproc is None:
        nproc = mp.cpu_count()
    fmt_print.bold_magenta(f'Running with {nproc} processes')

    process = {}

    if not lazy_iter:
        args = list(args)
        if kwargs is not None:
            kwargs = list(kwargs)
        else:
            kwargs = [{}] * len(args)
    else:
        raise NotImplementedError('lazy_iter=True is not supported, now')

    if len(args) != len(kwargs):
        raise ValueError('the length of args and kwargs must match !!!')

    p_bar = tqdm(total=len(args), desc=desc)

    results = {}
    count = 0
    while args or process:
        if len(process) < nproc and args:
            arg = args.pop()
            kw = kwargs.pop()

            q = mp.Queue()
            p = mp.Process(
                target=decorator(func),
                args=(q,) + arg,
                kwargs=kw
            )

            p.start()
            process[p] = q, time.time()

            q.count = count
            count += 1

        if process:
            to_remove = []
            for p, (q, t) in process.items():
                try:
                    results[q.count] = q.get(block=False)
                    p.terminate()

                    to_remove.append(p)

                except mp.queues.Empty:
                    if timeout and time.time() - t > timeout:
                        p.terminate()
                        if error_to_None:
                            results[q.count] = None
                            print(RuntimeWarning("Process {} timed out".format(q.count)))
                        else:
                            raise TimeoutError("The running process is timed out!!")

                        to_remove.append(p)

            for p in to_remove:
                del process[p]
                p_bar.update()

        if branch_jobs is not None:
            for branch_job in branch_jobs:
                branch_job(process, results, count)

        if isinstance(early_exit_condition, Callable) and early_exit_condition(results, count):
            for p, (q, t) in process.items():
                try:
                    results[q.count] = q.get(block=False)
                    p.terminate()
                except mp.queues.Empty:
                    p.kill()
                    p.terminate()
            while process:
                p, (q, t) = process.popitem()
                del p, q

            fmt_print.bold_magenta('MultiProcess Early Exit!!!')
            return [results[c] for c in sorted(results)]

    return [results[c] for c in sorted(results)]


class Pool:
    def __init__(
            self,
            nproc: int = None,
            desc: str = '',
            timeout: Optional[float] = None,
            error_to_None: bool = True
    ):
        self.nproc = nproc
        self.desc = desc
        self.timeout = timeout
        self.error_to_None = error_to_None

    def run(self, func: Callable, list_args: list[tuple], list_kwargs: list[dict]=None):
        return mp_run(func, list_args, list_kwargs, self.nproc, self.desc, self.timeout, self.error_to_None)
