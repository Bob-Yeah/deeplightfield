import sys
import time
import os

bar_length = 50
LAST_T = time.time()
BEGIN_T = LAST_T


def get_terminal_columns():
    return os.get_terminal_size().columns


def progress_bar(current, total, msg=None, premsg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.
    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    str0 = f"{premsg} [" if premsg else '['
    str1 = f"] {current + 1:d}/{total:d} | Step: {format_time(step_time)} | Tot: {format_time(total_time)}"
    if msg:
        str1 += f" | {msg}"

    tot_cols = get_terminal_columns()
    bar_length = tot_cols - len(str0) - len(str1)
    current_len = int(bar_length * (current + 1) / total)
    rest_len = int(bar_length - current_len)

    if current_len == 0:
        str_bar = '.' * rest_len
    else:
        str_bar = '=' * (current_len - 1) + '>' + '.' * rest_len

    sys.stdout.write(str0 + str_bar + str1)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# return the formatted time
def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds * 1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += '%02ds' % seconds_final
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += '%03dms' % millis
        time_index += 1
    if output == '':
        output = '0ms'
    return output
