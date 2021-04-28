import readline
import sys
import tty
import termios
import select
from typing import List


def make_completer(vocabulary):
    def custom_complete(text, state):
        #print('custom_complete: ', text, state)
        # None is returned for the end of the completion session.
        results = [x for x in vocabulary if x.startswith(text)] + [None]
        # A space is added to the completion since the Python readline doesn't
        # do this on its own. When a word is fully completed we want to mimic
        # the default readline library behavior of adding a space after it.
        return results[state] + " "
    return custom_complete


def check_input_in_list(value, list, err_msg):
    if value not in list:
        raise ValueError('Wrong input. (%s)' % err_msg)
    return value


def check_input_not_empty(value):
    if not value:
        raise ValueError('Wrong input. (Cannot be empty)')
    return value


def input_to_int(*, min=None, max=None):
    def action(s):
        try:
            s = int(s)
        except ValueError:
            raise ValueError('Wrong input. (Must be an integer)')
        if min != None and s < min:
            raise ValueError('Wrong input. (Must be larger than or equal to %d)' % min)
        if max != None and s > max:
            raise ValueError('Wrong input. (Must be less than or equal to %d)' % max)
        return s
    return action


def input_ex(prompt, *actions, default=None):
    prompt_default = '(Default: %s) ' % default if default != None else ''
    while True:
        s = input(prompt + ' ' + prompt_default).strip()
        try:
            if default == None:
                s = check_input_not_empty(s)
            if not s:
                s = default
            else:
                for action in actions:
                    s = action(s)
            break
        except ValueError as err:
            print(err.strerror)
    return s


def input_enum(prompt, complete_list: List[str], *, err_msg: str, default=None):
    readline.set_completer(make_completer(complete_list))
    prompt_default = '(Default: %s) ' % default if default != None else ''
    while True:
        s = input(prompt + ' ' + prompt_default).strip()
        try:
            if default == None:
                s = check_input_not_empty(s)
            s = check_input_in_list(
                s, complete_list, err_msg) if s else default
            break
        except ValueError as err:
            print(err)
    readline.set_completer()  # Clear completer
    return s


def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        if sys.stdin.readable:
            ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    select.select()
    return ch


def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)


if 'libedit' in readline.__doc__:  # mac OS/X 是走这个分支
    print('x')
    readline.parse_and_bind("bind ^I rl_complete")
else:
    readline.parse_and_bind("tab: complete")
readline.set_completer_delims(' ')
