from collections import Counter
from copy import copy


def get_counters(chart, filters):
    graph_sizes = Counter()
    for split in chart["START"]:
        assert len(split.items()) == 1
        if "max" in filters:
            graph_size = len(split["START"].nodeset)
            graph_sizes[graph_size] += 1
    ret = dict()
    if "max" in filters:
        ret["max"] = graph_sizes
    return ret


def filter_chart(chart, chart_filter, boundary_value):
    ret = copy(chart)
    derivations_to_keep = []
    for split in chart["START"]:
        assert len(split.items()) == 1
        if chart_filter == "max":
            graph_size = len(split["START"].nodeset)
            if graph_size >= boundary_value:
                derivations_to_keep.append(split)
    del ret["START"]
    ret["START"] = derivations_to_keep
    return ret


def get_filtered_chart(chart, chart_filter, counters):
    if chart_filter == "max":
        return filter_chart(chart, chart_filter, sorted(counters[chart_filter].items())[-1][0])
    else:
        return copy(chart)
