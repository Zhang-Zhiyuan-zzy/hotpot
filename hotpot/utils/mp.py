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
from typing import Iterable, Callable, Optional
import multiprocessing as mp


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
        error_to_None: bool = True
):

    if nproc is None:
        nproc = mp.cpu_count() // 2

    process = {}
    args = list(args)
    if kwargs is not None:
        kwargs = list(kwargs)
    else:
        kwargs = [{}] * len(args)

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

                # if not p.is_alive():
                #     results[q.count] =  q.get()
                #     p.terminate()
                #
                #     to_remove.append(p)
                #
                # elif timeout and time.time() - t > timeout:
                #     p.terminate()
                #     if error_to_None:
                #         results[q.count] = None
                #         print(RuntimeWarning("Process {} timed out".format(p.count)))
                #     else:
                #         raise TimeoutError("The running process is timed out!!")
                #
                #     to_remove.append(p)

            for p in to_remove:
                del process[p]
                p_bar.update()


    return [results[c] for c in sorted(results)]

