import torch.cuda


class Perf(object):

    def __init__(self, enable, start=False) -> None:
        super().__init__()
        self.enable = enable
        self.start_event = None
        if start:
            self.start()

    def start(self):
        if not self.enable:
            return
        if self.start_event == None:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self.start_event.record()

    def checkpoint(self, name: str = None, end: bool = False):
        if not self.enable:
            return 0
        self.end_event.record()
        torch.cuda.synchronize()
        duration = self.start_event.elapsed_time(self.end_event)
        if name:
            print('%s: %.1fms' % (name, duration))
        if not end:
            self.start_event.record()
        return duration
