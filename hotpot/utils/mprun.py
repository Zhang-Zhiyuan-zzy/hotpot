"""
python v3.9.0
@Project: hotpot
@File   : mprun
@Auther : Zhiyuan Zhang
@Data   : 2024/1/25
@Time   : 19:58
"""
import sys
import time
from tqdm import tqdm
from copy import copy
from typing import Callable
from multiprocessing import Process, Queue, cpu_count


class MpRun:
    """
    Running process with multiprocess
    Examples:
        def process(t: int, *, proc_name):
            time.sleep(t)
            print(f'finish process {proc_name} after {t} seconds')

        wait_times = {
            'proc1': ((1,), {'proc_name': 'first'}),
            'proc2': ((5,), {'proc_name': 'second'}),
            'proc3': ((10,), {'proc_name': 'third'})
        }
        mprun = MpRun(process)
        mprun.running(wait_times)

        for proc, res in mprun.results.item:
            print(proc, res)
    """
    def __init__(self, target: Callable, num_proc: int = None, wait_time: int = 10):
        self.target = target
        self.processes = {}
        self.num_proc = num_proc if isinstance(num_proc, int) else \
            cpu_count() // 2 if sys.platform == 'win32' else cpu_count()
        self.results = {}
        self.wait_time = wait_time

        self._p_bar = None

    def _get_results(self):
        for proc, queue in copy(self.processes).items():
            if not proc.is_alive():
                self.results[proc.name] = queue.get()
                proc.terminate()
                self.processes.pop(proc)
                self._p_bar.update()

    def _start_process(self, name, *args, **kwargs):
        queue = Queue()
        proc = Process(name=name, target=self._target, args=(queue,) + args, kwargs=kwargs)
        proc.start()
        self.processes[proc] = queue

    def _target(self, queue: Queue, *args, **kwargs):
        queue.put(self.target(*args, **kwargs))

    def running(self, in_stream: dict[str, tuple[tuple, dict]]):
        """
        running processes by given input stream
        Args:
            in_stream: a structured dict like: { process_name: ( (args,), {kwargs} ) }
        """
        self._p_bar = tqdm(in_stream, 'Running processes')
        while in_stream:
            if len(self.processes) < self.num_proc:
                name, (args, kwargs) = in_stream.popitem()
                self._start_process(name, *args, **kwargs)
            else:
                time.sleep(self.wait_time)

            self._get_results()

        while self.processes:
            self._get_results()


# Example
if __name__ == '__main__':
    def process(t: int, *, proc_name):
        count = 0
        while count < t * 1e7:
            count += 1
        return t, t*2, t*t, count

    import random
    wait_t = {f'proc{i}': ((random.randint(0, 20),), {'proc_name': f'Process {i}'}) for i in range(100)}

    mprun = MpRun(process, wait_time=1)
    mprun.running(wait_t)

    for proc, res in mprun.results.items():
        print(proc, res)
