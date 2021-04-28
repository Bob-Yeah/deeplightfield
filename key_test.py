import sys
import tty
import termios
import select
import time


def readchar():
    r, w, e = select.select([sys.stdin], [], [])
    if sys.stdin in r:
        ch = sys.stdin.read(1)
    return ch


fd = sys.stdin.fileno()
oldtty = termios.tcgetattr(fd)
newtty = termios.tcgetattr(fd)
try:
    termios.tcsetattr(fd, termios.TCSANOW, newtty)
    tty.setraw(fd)
    tty.setcbreak(fd)
    while True:
        print('Wait')
        time.sleep(0.1)
        key = readchar()
        print('%d' % ord(key))
        if key == 'w':
            print('w')
        if key == 'q':
            break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, oldtty)