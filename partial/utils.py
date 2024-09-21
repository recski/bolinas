import os


def get_range(in_dir, first, last):
    sen_dirs = sorted([int(d) for d in os.listdir(in_dir)])
    if first is None or first < sen_dirs[0]:
        first = sen_dirs[0]
    if last is None or last > sen_dirs[-1]:
        last = sen_dirs[-1]
    return [n for n in sen_dirs if first <= n <= last]