#!/usr/bin/env python2

import json
import os.path
import fileinput
import math
from argparse import ArgumentParser

from collections import Counter, defaultdict
from copy import copy

from common.hgraph.hgraph import Hgraph
from common import log
from common import output
from common.exceptions import DerivationException
from common.grammar import Grammar
from common.oie import get_labels
from parser.parser import Parser
from parser.vo_rule import VoRule
from parser_td.td_rule import TdRule
from parser_td.td_item import Item
from parser_td.parser_td import ParserTD


def get_range(in_dir, first, last):
    sen_dirs = sorted([int(d) for d in os.listdir(in_dir)])
    if first is None or first < sen_dirs[0]:
        first = sen_dirs[0]
    if last is None or last > sen_dirs[-1]:
        last = sen_dirs[-1]
    return range(first,  last + 1)


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


def get_rules(derivation):
    if type(derivation) is not tuple:
        if derivation == "START":
            return {}
        return {derivation.rule.rule_id: str(derivation.rule)}
    else:
        ret = {}
        items = [c for (_, c) in derivation[1].items()] + [derivation[0]]
        for item in items:
            for (k, v) in get_rules(item).items():
                assert k not in ret or v == ret[k]
                ret[k] = v
        return ret


def main(in_dir, first, last, grammar_file, chart_filters, parser_type, boundary_nodes, k):
    logprob = False
    nodelabels = True
    backward = False

    for filter in chart_filters:
        assert filter in ['basic', 'max', 'prec', 'rec']

    with open(grammar_file, 'ra') as f:
        if parser_type == 'td':
            parser_class = ParserTD
            rule_class = TdRule
            if boundary_nodes:
                parser_class.item_class = Item
        elif parser_type == 'basic':
            parser_class = Parser
            rule_class = VoRule
        grammar = Grammar.load_from_file(f, rule_class, backward, nodelabels=nodelabels, logprob=logprob)
    log.info("Loaded %s%s grammar with %i rules." % (grammar.rhs1_type, "-to-%s" % grammar.rhs2_type if grammar.rhs2_type else '', len(grammar)))

    parser = parser_class(grammar)

    for i in get_range(in_dir, first, last):
        print "\nProcessing sen %d\n" % i
        sen_dir = os.path.join(in_dir, str(i))

        preproc_dir = os.path.join(sen_dir, "preproc")
        graph_file = os.path.join(preproc_dir, "sen" + str(i) + ".graph")
        with open(os.path.join(preproc_dir, "sen" + str(i) + "_pa_nodes.json")) as f:
            pa_nodes = json.load(f)

        bolinas_dir = os.path.join(sen_dir, "bolinas")
        if not os.path.exists(bolinas_dir):
            os.makedirs(bolinas_dir)
        match_file = os.path.join(bolinas_dir, "sen" + str(i) + "_matches.graph")
        labels_file = os.path.join(bolinas_dir, "sen" + str(i) + "_predicted_labels.txt")
        rules_file = os.path.join(bolinas_dir, "sen" + str(i) + "_derivation.txt")

        parse_generator = parser.parse_graphs(
            (Hgraph.from_string(x) for x in fileinput.input(graph_file)), partial=True, max_steps=10000)
    
        for chart in parse_generator:
            if "START" not in chart:
                print "No derivation found"
                continue
            matches_lines = []
            labels_lines = []
            rules_lines = []
            matches_output = defaultdict(list)
            counters = get_counters(chart, pa_nodes, chart_filters)
            for chart_filter in chart_filters:
                if chart_filter == "max":
                    filtered_chart = filter_chart(chart, pa_nodes, chart_filter, sorted(counters[chart_filter].items())[-1][0])
                elif chart_filter == "prec":
                    filtered_chart = filter_chart(chart, pa_nodes, chart_filter, sorted(counters[chart_filter].items())[0][0])
                    counter = get_counters(filtered_chart, pa_nodes, ["rec"])
                    filtered_chart = filter_chart(filtered_chart, pa_nodes, "rec", sorted(counter["rec"].items())[-1][0])
                elif chart_filter == "rec":
                    filtered_chart = filter_chart(chart, pa_nodes, chart_filter, sorted(counters[chart_filter].items())[-1][0])
                    counter = get_counters(filtered_chart, pa_nodes, ["prec"])
                    filtered_chart = filter_chart(filtered_chart, pa_nodes, "prec", sorted(counter["prec"].items())[0][0])
                else:
                    filtered_chart = copy(chart)
                
                kbest = filtered_chart.kbest('START', k)
                if kbest and kbest < k:
                    log.info("Found only %i derivations." % len(kbest))
                matches_lines.append("%s\n" % chart_filter)
                labels_lines.append("%s\n" % chart_filter)
                rules_lines.append("%s\n" % chart_filter)
                for score, derivation in kbest:
                    n_score = score if logprob else math.exp(score)
                    try:
                        shifted_derivation = output.print_shifted(derivation)
                        matches_output[chart_filter].append(shifted_derivation)
                        format_derivation = output.format_derivation(derivation)
                        labels = get_labels(derivation)
                        rules = get_rules(derivation)
                        
                        matches_lines.append("%s\n" % shifted_derivation)
                        labels_lines.append("%s\n" % json.dumps(labels))
                        rules_lines.append("%s\n" % format_derivation)
                        for grammar_nr, rule_str in sorted(rules.items()):
                            prob = rule_str.split(';')[1].strip()
                            rule = rule_str.split(';')[0].strip()
                            rules_lines.append("%s\t%.2f\t%s\n" % (grammar_nr, float(prob), rule))
                        rules_lines.append("\n")
 
                        print "\n%s" % chart_filter
                        print "%s\t#%g" % (format_derivation, n_score)
                        print "%s\t#%g" % (output.apply_graph_derivation(derivation).to_string(newline=False), n_score)
                        print "%s" % shifted_derivation
                    except DerivationException, e:
                        log.err("Could not construct derivation: '%s'. Skipping." % e.message)
            with open(match_file, "w") as f:
                f.writelines(matches_lines)
            with open(labels_file, "w") as f:
                f.writelines(labels_lines)
            with open(rules_file, "w") as f:
                f.writelines(rules_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description ="Bolinas is a toolkit for synchronous hyperedge replacement grammars.")
    parser.add_argument("-g", "--grammar_file", help="A hyperedge replacement grammar (HRG) or synchronous HRG (SHRG).")
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    parser.add_argument('-c', '--chart-filters', nargs='+', default=["basic"], help="A list of 'basic' (no filters), 'max' (biggest overlaps), 'prec' (best precision) or 'rec' (best recall). 'prec' and 'rec' only works if gold data is provided.")
    parser.add_argument("-ot", "--output_type", type=str, default="derived", help="Set the type of the output to be produced for each object in the input file. \n'forest' produces parse forests.\n'derivation' produces k-best derivations.\n'derived' produces k-best derived objects (default).")
    parser.add_argument("-k", type=int, default=1, help ="Generate K best derivations for the objects in the input file. Cannot be used with -g (default with K=1).")
    parser.add_argument("-p", "--parser", default="basic", help="Specify which graph parser to use. 'td': the tree decomposition parser of Chiang et al, ACL 2013. 'basic': a basic generalization of CKY that matches rules according to an arbitrary visit order on edges (less efficient).")
    parser.add_argument("-bn", "--boundary_nodes", action="store_true", help="In the tree decomposition parser, use the full representation for graph fragments instead of the compact boundary node representation. This can provide some speedup for grammars with small rules.")
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
         args.chart_filters,
         args.parser,
         args.boundary_nodes,
         args.k)