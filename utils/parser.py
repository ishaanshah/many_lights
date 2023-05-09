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

def add_test_args(parser: ArgumentParser):
    parser.add_argument("--slow", action="store_true")