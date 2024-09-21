#!/usr/bin/env python2

import json
import os.path
import fileinput
import math
from argparse import ArgumentParser

from collections import Counter
from copy import copy

from common.hgraph.hgraph import Hgraph
from common import log
from common.exceptions import DerivationException
from common.grammar import Grammar
from common.oie import get_labels
from parser_basic.parser import Parser
from parser_basic.vo_rule import VoRule


def get_range(in_dir, first, last):
    sen_dirs = sorted([int(fn.split(".")[0]) for fn in os.listdir(in_dir)])
    if first is None or first < sen_dirs[0]:
        first = sen_dirs[0]
    if last is None or last > sen_dirs[-1]:
        last = sen_dirs[-1]
    return range(first,  last + 1)


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


def main(in_dir, first, last, grammar_file, out_dir):
    logprob = False
    nodelabels = True
    backward = False
    k = 1
    chart_filter = "max"

    with open(grammar_file, 'ra') as f:
        parser_class = Parser
        rule_class = VoRule
        grammar = Grammar.load_from_file(f, rule_class, backward, nodelabels=nodelabels, logprob=logprob)
    log.info("Loaded %s%s grammar with %i rules." % (grammar.rhs1_type, "-to-%s" % grammar.rhs2_type if grammar.rhs2_type else '', len(grammar)))

    parser = parser_class(grammar)

    for sen_id in get_range(in_dir, first, last):
        if first is not None and sen_id < first:
            continue
        if last is not None and sen_id > last:
            continue

        print "\nProcessing sen %d\n" % sen_id

        graph_file = os.path.join(in_dir, str(sen_id) + ".graph")
        labels_file = os.path.join(out_dir, str(sen_id) + ".txt")

        parse_generator = parser.parse_graphs((Hgraph.from_string(x) for x in fileinput.input(graph_file)),
                                              partial=True, max_steps=10000)

        for chart in parse_generator:
            print "Chart size: %d" % len(chart)
            
            if "START" not in chart:
                print "No derivation found"
                continue
            
            labels_lines = []
            counters = get_counters(chart, [chart_filter])
            filtered_chart = filter_chart(chart, chart_filter, sorted(counters[chart_filter].items())[-1][0])

            kbest = filtered_chart.kbest('START', k)
            if kbest and kbest < k:
                log.info("Found only %i derivations." % len(kbest))
            labels_lines.append("%s\n" % chart_filter)
            for score, derivation in kbest:
                n_score = score if logprob else math.exp(score)
                try:
                    labels = get_labels(derivation)
                    labels_lines.append("%s\n" % json.dumps(labels))
                except DerivationException, e:
                    log.err("Could not construct derivation: '%s'. Skipping." % e.message)
            with open(labels_file, "w") as f:
                f.writelines(labels_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description ="Bolinas is a toolkit for synchronous hyperedge replacement grammars.")
    parser.add_argument("-g", "--grammar_file", help="A hyperedge replacement grammar (HRG) or synchronous HRG (SHRG).")
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    parser.add_argument("-v", "--verbose", type=int, default=2, help="Stderr output verbosity: 0 (all off), 1 (warnings), 2 (info, default), 3 (details), 3 (debug)")

    args = parser.parse_args()

    # Definition of logger output verbosity levels 
    log.LOG = {0:{log.err},
               1:{log.err, log.warn},
               2:{log.err, log.warn, log.info},
               3:{log.err, log.warn, log.info, log.chatter},
               4:{log.err, log.warn, log.chatter, log.info, log.debug}
              }[args.verbose]

    main(args.in_dir,
         args.first,
         args.last,
         args.grammar_file,
         args.out_dir)
