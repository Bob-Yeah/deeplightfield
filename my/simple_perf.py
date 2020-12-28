import torch.cuda


class SimplePerf(object):

    def __init__(self, enable, start = False) -> None:
        super().__init__()
        self.enable = enable
        self.start_event = None
        if start:
            self.Start()
    
    def Start(self):
        if not self.enable:
            return
        if self.start_event == None:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self.start_event.record()
    
    def Checkpoint(self, name: str, end: bool = False):
        if not self.enable:
            return
        self.end_event.record()
        torch.cuda.synchronize()
        print('%s: %.1fms' % (name, self.start_event.elapsed_time(self.end_event)))
        if not end:
            self.start_event.record()