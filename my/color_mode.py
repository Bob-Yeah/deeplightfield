RGB = 0
GRAY = 1
YCbCr = 2


def to_str(color_mode):
    return "gray" if color_mode == GRAY \
        else ("ybr" if color_mode == YCbCr
              else "rgb")


def from_str(color_str):
    return GRAY if color_str == 'gray' \
        else (YCbCr if color_str == 'ybr'
              else RGB)
