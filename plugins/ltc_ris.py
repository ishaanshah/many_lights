from typing import Tuple
import mitsuba as mi
mi.set_variant("cuda_rgb", "llvm_rgb")
mi.set_log_level(mi.LogLevel.Info)

import drjit as dr
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from utils.render import render_multi_pass, linear_to_srgb
from utils.parser import add_common_args
from utils.reservoir import Reservoir
from utils.fs import create_dir
from utils.ltc import si_ltc

EPS = dr.epsilon(mi.Float)

class LTC_RISIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)

        self.m = props.get("num_proposals", 32)
        self.p = props.get("num_pdf_samples", 4)

        self.ltc_1: mi.Texture = props["ltc_1"]
        self.ltc_2: mi.Texture = props["ltc_2"]
        self.ltc_3: mi.Texture = props["ltc_3"]

    def estimate_pdf(self, si: mi.SurfaceInteraction3f, emitter: mi.Emitter, sampler: mi.Sampler, active: bool=True) -> mi.Float:
        """ Estimate the PDF of choosing a given emitter """
        bsdf: mi.BSDF = si.bsdf()
        ctx: mi.BSDFContext = mi.BSDFContext()

        p = mi.Float(0)
        i = mi.UInt(0)
        loop = mi.Loop("PDF Integration", lambda: (p, i, sampler))
        while loop(i < self.p):
            ds, emitter_val = emitter.sample_direction(si, sampler.next_1d(active), active)

            # Evaluate BSDF
            wo = si.to_local(ds.d)
            bsdf_val: mi.Color3f = bsdf.eval(ctx, si, wo, active)

            p += dr.norm(bsdf_val * emitter_val)
            i += 1

        return p

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
        i = mi.UInt(0)
        reservoir = Reservoir(mi.UInt)
        loop = mi.Loop("RIS", lambda: (i, reservoir.sample, reservoir.w_sum, sampler, reservoir.pdf))
        while loop(i < self.m):
            # Sample an emitter
            emitter_idx, inv_pdf, _ = scene.sample_emitter(sampler.next_1d(active), active)
            emitter = dr.gather(type(emitters), emitters, emitter_idx, active)

            # Calculate RIS weight and update reservoir
            p_hat = self.estimate_pdf(si, emitter, sampler, active)
            w = p_hat * inv_pdf
            reservoir.update(w, emitter_idx, sampler.next_1d(active), p_hat)

            i += 1

        emitter: mi.Emitter = dr.gather(type(emitters), emitters, reservoir.sample, active)

        # TODO: Perform LTC integration
        ltc_result = emitter.shape().eval_ltc(si, 0, active)
        new_var = ltc_result.y

        # p_hat = self.estimate_pdf(si, emitter, sampler, active)
        p_hat = reservoir.pdf
        W = dr.select(p_hat > EPS, dr.rcp(p_hat) * reservoir.w_sum * dr.rcp(self.m), 0)

        result += dr.select(active, new_var * W, 0) * emitter.sample_wavelengths(si, 0, active)[1]

        return result, active, []