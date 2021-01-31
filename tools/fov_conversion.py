import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--fov', type=float,
                    help='Field of view')
parser.add_argument('--fov0', type=float, required=True,
                    help='Full screen field of view')
parser.add_argument('--res0', type=int, required=True,
                    help='Full screen resolution')
opt = parser.parse_args()


if __name__ == "__main__":
    if opt.fov:
        pixels = math.tan(math.radians(opt.fov / 2)) / \
            math.tan(math.radians(opt.fov0 / 2)) * opt.res0
        print('Pixels of FOV %.1f is %.1f' % (opt.fov, pixels))
    if opt.pixels:
        fov = math.degrees(math.atan(opt.pixels / opt.res0 *
                                     math.tan(math.radians(opt.fov0 / 2)))) * 2
        print('FOV of pixels %.1f is %.1f' % (opt.pixels, fov))
