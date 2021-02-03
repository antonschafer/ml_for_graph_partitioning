import time
from collections import defaultdict


class Timer:
    def __init__(self):
        self.times = defaultdict(lambda:0)
        self.start_time = time.time()
        self.last_op_end_time = time.time()
        self.current_op = None

    def reset(self):
        self.times = defaultdict(lambda:0)
        self.start_time = time.time()
        self.last_op_end_time = time.time()
        self.current_op = None

    def total(self):
        return time.time() - self.start_time

    def enter_op(self, op):
        if self.current_op is not None:
            self.times[self.current_op] += time.time() - self.last_op_end_time

        self.current_op = op
        self.last_op_end_time = time.time()

    def done(self):
        if self.current_op is not None:
            self.times[self.current_op] += time.time() - self.last_op_end_time

        self.current_op = None
        self.last_op_end_time = time.time()

    def get_time(self, op):
        if self.current_op == op:
            raise Exception("op still running")
        return self.times[op]

    def report(self):
        res_string = "Time total: {:.4f}".format(self.total())
        for op, t in self.times.items():
            res_string += ", Time " + op + ": {:.4f}".format(t)
        return res_string



