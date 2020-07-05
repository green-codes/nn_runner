import multiprocessing
import os


class MultiProcessor:
    """
    Simple, general-purpose parallel processing class, chiefly intended for
    parallel file processing. Suitable for CPU-bottlenecked operations only.
    Takes an array of input data (typically a list of file paths) and a
    callable transform object, then maps the transform over the array.
    """

    def __init__(self, transform, arr, num_workers=0):
        num_workers = num_workers if num_workers > 0 else os.cpu_count()
        num_workers = min(len(arr), num_workers)

        # create subarrays
        assert isinstance(arr, list)
        s_len = (len(arr) // num_workers) + int(len(arr) % num_workers > 0)
        self.s_arr = [arr[i*s_len:(i+1)*s_len] for i in range(num_workers)]
        print(s_len)

        # process manager for progress reports and results
        self.manager = multiprocessing.Manager()
        self.stats = self.manager.dict()  # shared dict for progress reports
        self.res = self.manager.dict()    # shared dict for returned results

        # create processes
        self.p_arr = [multiprocessing.Process(
            target=MultiProcessor._transform_runner,
            args=(transform, self.s_arr[i], self.stats, self.res, i))
            for i in range(num_workers)]

    def run(self):
        """ Start all processes """
        for i, p in enumerate(self.p_arr):
            self.stats[i] = 0
            p.start()

    def join(self):
        """ Wait for all processes to complete """
        for p in self.p_arr:
            p.join()

    def terminate(self):
        """ Terminate all processes """
        for p in self.p_arr:
            p.terminate()

    def status_report(self):
        """ returns completed and total samples for each process """
        stats = [(self.stats[i], len(self.s_arr[i]))
                 for i in range(len(self.p_arr))]
        return stats

    def get_pids(self):
        return [p.pid for p in self.p_arr]

    @staticmethod
    def _transform_runner(transform, arr, stats, res, i):
        for j, e in enumerate(arr):
            ret = transform(e)
            if i in res:
                res[i] += [ret]  # do transform & record results
            else:
                res[i] = [ret]
            stats[i] = j + 1    # report progress
