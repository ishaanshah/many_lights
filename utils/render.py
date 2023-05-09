import mitsuba as mi
import drjit as dr
import time
import numpy as np
from typing import Tuple, Callable

def linear_to_srgb(image: mi.TensorXf|np.ndarray, gamma: float=2.2) -> mi.TensorXf|np.ndarray:
    if isinstance(image, np.ndarray):
        return np.clip(image ** (1 / gamma), 0, 1)
    else:
        return dr.clamp(image ** (1 / gamma), 0, 1)

def get_spp_per_pass(res_x: int, res_y: int, spp: int) -> Tuple[int, int]:
    spp_per_pass = int(spp)
    samples_per_pass = res_x * res_y * spp_per_pass
    wavefront_size_limit = int(2**32)
    if samples_per_pass > wavefront_size_limit:
        spp_per_pass = spp_per_pass // int((samples_per_pass + wavefront_size_limit - 1) / wavefront_size_limit)
        n_passes = spp // spp_per_pass
        samples_per_pass = res_x * res_y * spp_per_pass
        mi.Log(mi.LogLevel.Warn, f"Too many samples requested, splitting the job into {n_passes} passes with {spp_per_pass} spp")
        return spp_per_pass, n_passes
    else:
        return spp_per_pass, 1

def render_multi_pass(render_func: Callable, res_x: int, res_y: int, scene: mi.Scene, spp: int, save_path: str=None) -> np.ndarray:
    t0 = time.time()
    samples_per_pass, n_passes = get_spp_per_pass(res_x, res_y, spp)
    for pass_ in range(n_passes):
        t0 = time.time()
        if pass_ == 0:
            result = render_func(scene=scene, spp=samples_per_pass, seed=pass_)
        else:
            result += render_func(scene=scene, spp=samples_per_pass, seed=pass_)
        dr.eval(result)

    result = result / n_passes

    if save_path:
        mi.util.write_bitmap(save_path, result)

    result = result.numpy() # Forces evaluation
    mi.Log(mi.LogLevel.Info, f"Time taken: {time.time() - t0:.2f}s", )

    return result

def render_manual(
    render_func: Callable[[mi.SurfaceInteraction3f], mi.Color3f],
    scene: mi.Scene,
    spp: int = None,
    random_offset: bool=True
) -> mi.TensorXf:
    # Sensor & Film
    sensor: mi.Sensor = scene.sensors()[0]
    film: mi.Film = sensor.film()
    film_size = film.crop_size() # [width, height] image size
    if film.sample_border():
        # For correctness, we need to sample extra pixels on the border
        # Otherwise, convolution will have black pixels at the border
        # film.rfilter().border_size() is mult. by 2 to account for left/top & right/bottom borders
        film_size += 2 * film.rfilter().border_size()

    sampler: mi.Sampler = sensor.sampler()
    if spp:
        sampler.set_sample_count(spp)
    spp = sampler.sample_count()
    film.prepare([])

    # Wavefront setup
    wavefront_size = film_size.x * film_size.y * spp

    sampler.set_samples_per_wavefront(spp) # There are 'spp' number of passes
    sampler.seed(0, wavefront_size)

    # Image block
    block = film.create_block()
    # Offset is the currect location of the block
    # In case of GPU, the block covers the entire image, hence offset is 0
    block.set_offset(film.crop_offset())
    # Coalescing is batch read/writes, useful for efficient memory accesses.
    block.set_coalesce(block.coalesce() and spp >= 4)

    idx = dr.arange(dr.cuda.UInt32, 0, wavefront_size)
    idx = idx // dr.cuda.UInt32(spp)

    pos = mi.Vector2u(0)
    pos.y = idx // film_size.x
    pos.x = -film_size.x * pos.y + idx

    if film.sample_border():
        pos = pos - film.rfilter().border_size()

    pos = pos + film.crop_offset()

    diff_scale_factor = dr.rsqrt(spp)

    ################################
    # Camera rays
    ################################
    scale = 1.0 / mi.ScalarVector2f(film.crop_size())
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale

    if random_offset:
        sample_pos = pos + sampler.next_2d()
    else:
        sample_pos = pos

    adjusted_pos = sample_pos * scale + offset # Float in range [0, 1] in each dimension [width, height]

    aperture_sample = mi.Point2f(0)
    if sensor.needs_aperture_sample():
        aperture_sample = sampler.next_2d()

    time = sensor.shutter_open()
    if sensor.shutter_open_time() > 0.0:
        time = time + sampler.next_1d() * sensor.shutter_open_time()
    
    wavelength_sample = mi.Float(0)

    ray, ray_weight = sensor.sample_ray_differential(
        time, wavelength_sample, adjusted_pos, aperture_sample
    )
    if ray.has_differentials:
        ray.scale_differential(diff_scale_factor)

    final_color = [mi.Float(0), mi.Float(0), mi.Float(0), mi.Float(1)]

    ################################
    # Rendering algorithm
    ################################
    si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
    res = render_func(si)
    final_color[0] = res.x
    final_color[1] = res.y
    final_color[2] = res.z

    ################################
    # Save image
    ################################
    block.put(sample_pos, final_color)
    film.put_block(block)
    img = film.develop()

    return img