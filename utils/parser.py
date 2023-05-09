from argparse import ArgumentParser

def add_common_args(parser: ArgumentParser):
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--output_dir", default="renders")
    parser.add_argument("--show_render", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--disable_bsdf", action="store_true")
    parser.add_argument("--disable_emitter", action="store_true")
    parser.add_argument("--gt_spp", type=int, default=50000)
    parser.add_argument("--spp", type=int, default=1000)
    parser.add_argument("-w", action="count", default=0)

def add_glint_args(parser: ArgumentParser):
    parser.add_argument("--ndf_texture", default="textures/scratch_wave_0.05.exr")
    parser.add_argument("--grid_size", default=100, type=int)
    parser.add_argument("--uv_scale", default=5, type=float)
    parser.add_argument("--method", choices=["naive", "hist", "sat_hist"], default="hist")
    parser.add_argument("--ntheta", type=int, default=9)
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--diffuse", default=[0.0, 0.0, 0.0], nargs=3)

def add_test_args(parser: ArgumentParser):
    parser.add_argument("--slow", action="store_true")

def add_sh_args(parser: ArgumentParser):
    parser.add_argument("--to_fit", type=str, default="envmaps/indoor_1.hdr")
    parser.add_argument("--samples", type=int, default=1e7)
    parser.add_argument("--order", type=int, default=9)