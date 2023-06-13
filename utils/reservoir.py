import mitsuba as mi
import drjit as dr

class Reservoir():
    def __init__(self, dtype):
        self.sample = dr.zeros(dtype)
        self.pdf = dr.zeros(mi.Float)
        self.w_sum = dr.zeros(mi.Float)
        self.dtype = dtype
    
    def update(self, w: mi.Float, x, rand: mi.Float, p):
        self.w_sum += w
        self.sample = dr.select(rand < w / self.w_sum, x, self.sample)
        self.pdf = dr.select(rand < w / self.w_sum, p, self.pdf)
    
    def __repr__(self) -> str:
        return f"w_sum: {self.w_sum}, sample: {self.sample}"