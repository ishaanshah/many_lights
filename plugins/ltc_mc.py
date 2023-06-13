from typing import Tuple
import mitsuba as mi
mi.set_variant("cuda_rgb", "llvm_rgb")
mi.set_log_level(mi.LogLevel.Info)

import drjit as dr
from utils.ltc import si_ltc

EPS = dr.epsilon(mi.Float)

class LTC_MCIntegragtor(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)

        self.ltc_1: mi.Texture = props["ltc_1"]
        self.ltc_2: mi.Texture = props["ltc_2"]
        self.ltc_3: mi.Texture = props["ltc_3"]

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True) -> Tuple[mi.Color3f, bool]:
        si = scene.ray_intersect(ray, active)

        bsdf: mi.BSDF = si.bsdf()
        bsdf_flags: mi.BSDFFlags = bsdf.flags()

        active = si.is_valid() & mi.has_flag(bsdf_flags, mi.BSDFFlags.Smooth)

        result = mi.Color3f(0)

        # Show emitters
        result += si.emitter(scene, active).eval(si, active)

        # Instantiate LTC stuff
        si_ltc(si, sampler, self.ltc_1, self.ltc_2, self.ltc_3)

        emitters = scene.emitters_dr()

        # Sample an emitter
        emitter_idx, inv_pdf, _ = scene.sample_emitter(sampler.next_1d(active), active)
        emitter: mi.Emitter = dr.gather(type(emitters), emitters, emitter_idx, active)

        shading_coeffs = emitter.shape().eval_ltc(si, 0, active=active)
        result += shading_coeffs.y * inv_pdf * emitter.sample_wavelengths(si, 0, active)[1]

        return result, active, []