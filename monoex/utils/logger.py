import functools
import logging
import os
import sys
from collections import Counter, defaultdict, deque

import torch


@functools.lru_cache()
def setup_logger(output_dir, distributed_rank=0, name="core", file_name="log.txt"):
    """
    Args:
        output_dir (str): a directory saves output log files
        name (str): name of the logger
        file_name (str): name of log file
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    if output_dir:
        fh = logging.FileHandler(os.path.join(output_dir, file_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "lumos"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()


def log_first_n(lvl, msg, n=1, *, name=None, key="caller"):
    """
    Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or "message", which 
            defines how to identify duplicated logs. For example, if called with `n=1, 
            key="caller"`, this function will only log the first call from the same caller, 
            regardless of the message content.
            If called with `n=1, key="message"`, this function will log the same content 
            only once, even if they are called from different places. If called with `n=1, 
            key=("caller", "message")`, this function will not log only if the same caller 
            has logged the same message before.
    """
    if isinstance(key, str):
        key = (key, )
    assert len(key) > 0

    caller_module, caller_key = _find_caller()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (msg, )

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(lvl, msg)


class SmoothedValue():
    """
    Track a series of values and provide access to smoothed values over 
    a window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def value(self):
        d = torch.tensor(list(self.deque))
        return d[-1].item()

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger():

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("{} object has no attribute {}".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.avg))
        return self.delimiter.join(loss_str)
