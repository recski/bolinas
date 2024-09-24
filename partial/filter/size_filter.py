from collections import Counter
from copy import copy


def get_size_counter(chart):
    graph_sizes = Counter()
    for split in chart["START"]:
        assert len(split.items()) == 1
        graph_size = len(split["START"].nodeset)
        graph_sizes[graph_size] += 1
    return graph_sizes


def delete_smaller(chart, boundary_value):
    ret = copy(chart)
    splits_to_keep = []
    for split in chart["START"]:
        assert len(split.items()) == 1
        graph_size = len(split["START"].nodeset)
        if graph_size >= boundary_value:
            splits_to_keep.append(split)
    del ret["START"]
    ret["START"] = splits_to_keep
    return ret


def filter_for_size(chart, chart_filter):
    boundary_value = 0
    if chart_filter == "max":
        size_counter = get_size_counter(chart)
        boundary_value = sorted(size_counter.keys(), reverse=True)[0]
    filtered_chart = delete_smaller(chart, boundary_value)
    return filtered_chart
