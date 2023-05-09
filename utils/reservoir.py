import mitsuba as mi
import drjit as dr

class Reservoir():
    def __init__(self, dtype, width: int, initseq: mi.UInt64, initstate: mi.UInt64):
        self.width = width
        self.rng = mi.PCG32(width, initseq=initseq, initstate=initstate)
        self.sample = dr.zeros(dtype, width)
        self.w_sum = dr.zeros(mi.Float, width)
        self.dtype = dtype
    
    def update(self, w: mi.Float, x, active: bool=True):
        rnd = self.rng.next_float32(active)
        self.w_sum += w
        self.sample = self.dtype(dr.select((rnd < w / dr.maximum(self.w_sum, 1e-6)), x, self.sample))
    
    def __repr__(self) -> str:
        return f"w_sum: {self.w_sum}, sample: {self.sample}"