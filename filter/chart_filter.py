from collections import Counter
from copy import copy


def get_counters(chart, pa_nodes, filters):
    graph_sizes = Counter()
    intersect_sizes = Counter()
    diff_sizes = Counter()
    for split in chart["START"]:
        assert len(split.items()) == 1
        if "max" in filters:
            graph_size = len(split["START"].nodeset)
            graph_sizes[graph_size] += 1
        if "prec" in filters:
            diff = set(split["START"].nodeset) - set(pa_nodes)
            diff_size = len(diff)
            diff_sizes[diff_size] += 1
        if "rec" in filters:
            intersect = set(pa_nodes) & set(split["START"].nodeset)
            intersect_size = len(intersect)
            intersect_sizes[intersect_size] += 1
    ret = dict()
    if "max" in filters:
        ret["max"] = graph_sizes
    if "prec" in filters:
        ret["prec"] = diff_sizes
    if "rec" in filters:
        ret["rec"] = intersect_sizes
    return ret


def filter_chart(chart, pa_nodes, chart_filter, boundary_value):
    ret = copy(chart)
    derivations_to_keep = []
    for split in chart["START"]:
        assert len(split.items()) == 1
        if chart_filter == "max":
            graph_size = len(split["START"].nodeset)
            if graph_size >= boundary_value:
                derivations_to_keep.append(split)
        elif chart_filter == "prec":
            diff = set(split["START"].nodeset) - set(pa_nodes)
            diff_size = len(diff)
            if diff_size <= boundary_value:
                derivations_to_keep.append(split)
        elif chart_filter == "rec":
            intersect = set(pa_nodes) & set(split["START"].nodeset)
            intersect_size = len(intersect)
            if intersect_size >= boundary_value:
                derivations_to_keep.append(split)
    del ret["START"]
    ret["START"] = derivations_to_keep
    return ret


def get_filtered_chart(chart, pa_nodes, chart_filter, counters):
    if chart_filter == "max":
        return filter_chart(chart, pa_nodes, chart_filter, sorted(counters[chart_filter].items())[-1][0])
    elif chart_filter == "prec":
        filtered_chart = filter_chart(chart, pa_nodes, chart_filter, sorted(counters[chart_filter].items())[0][0])
        counter = get_counters(filtered_chart, pa_nodes, ["rec"])
        return filter_chart(filtered_chart, pa_nodes, "rec", sorted(counter["rec"].items())[-1][0])
    elif chart_filter == "rec":
        filtered_chart = filter_chart(chart, pa_nodes, chart_filter, sorted(counters[chart_filter].items())[-1][0])
        counter = get_counters(filtered_chart, pa_nodes, ["prec"])
        return filter_chart(filtered_chart, pa_nodes, "prec", sorted(counter["prec"].items())[0][0])
    else:
        return copy(chart)
