#!/usr/bin/env python2
import json
import os.path
import fileinput
import pickle
import time

from argparse import ArgumentParser
from datetime import datetime

from common.hgraph.hgraph import Hgraph
from common import log
from common.grammar import Grammar
from parser_basic.parser import Parser
from parser_basic.vo_rule import VoRule
from parser_td.td_rule import TdRule
from parser_td.td_item import Item
from parser_td.parser_td import ParserTD
from partial.utils import get_range


def load_grammar(grammar_file, parser_type, boundary_nodes, backward, nodelabels, logprob):
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
    log.info("Loaded %s%s grammar with %i rules." %
             (grammar.rhs1_type, "-to-%s" % grammar.rhs2_type if grammar.rhs2_type else '', len(grammar)))
    return grammar, parser_class


def parse_sen(graph_parser, graph_file, chart_file):
    parse_generator = graph_parser.parse_graphs(
        (Hgraph.from_string(x) for x in fileinput.input(graph_file)), partial=True, max_steps=10000)
    for i, chart in enumerate(parse_generator):
        assert i == 0
        if "START" not in chart:
            print "No derivation found"
            continue
        else:
            print "Chart len: %d" % len(chart)
            with open(chart_file, "w") as f:
                pickle.dump(chart, f)


def main(data_dir, config_json, grammar_file, parser_type, boundary_nodes):
    start_time = time.time()
    logprob = True
    nodelabels = True
    backward = False
    config = json.load(open(config_json))

    log_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "log",
        "parse_" + config["out_dir"] + ".log"
    )
    log_lines = ["Execution start: %s\n" % str(datetime.now())]
    first = config.get("first", None)
    last = config.get("last", None)

    grammar, parser_class = load_grammar(grammar_file, parser_type, boundary_nodes, backward, nodelabels, logprob)

    graph_parser = parser_class(grammar)

    in_dir = os.path.join(data_dir, config["preproc_dir"])
    out_dir = os.path.join(data_dir, config["out_dir"])
    for sen_idx in get_range(in_dir, first, last):
        print "\nProcessing sen %d\n" % sen_idx
        sen_dir_in = os.path.join(in_dir, str(sen_idx))
        sen_dir_out = os.path.join(out_dir, str(sen_idx))
        preproc_dir = os.path.join(sen_dir_in, "preproc")
        graph_file = os.path.join(preproc_dir, "pos_edge.graph")

        bolinas_dir = os.path.join(sen_dir_out, "bolinas")
        if not os.path.exists(bolinas_dir):
            os.makedirs(bolinas_dir)
        chart_file = os.path.join(bolinas_dir, "sen" + str(sen_idx) + "_chart.pickle")

        parse_sen(graph_parser, graph_file, chart_file)


    log_lines.append("Execution finish: %s\n" % str(datetime.now()))
    elapsed_time = time.time() - start_time
    time_str = "Elapsed time: %d min %d sec" % (elapsed_time / 60, elapsed_time % 60)
    print time_str
    log_lines.append(time_str)
    with open(log_file, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description ="Parse graph inputs and save chart.")
    parser.add_argument("-g", "--grammar_file",
                        help="A hyperedge replacement grammar (HRG) or synchronous HRG (SHRG).")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-ot", "--output_type", type=str, default="derived",
                        help="Set the type of the output to be produced for each object in the input file. \n"
                             "'forest' produces parse forests.\n'derivation' produces k-best derivations.\n"
                             "'derived' produces k-best derived objects (default).")
    parser.add_argument("-k", type=int, default=1,
                        help="Generate K best derivations for the objects in the input file. "
                             "Cannot be used with -g (default with K=1).")
    parser.add_argument("-p", "--parser", default="basic",
                        help="Specify which graph parser to use. "
                             "'td': the tree decomposition parser of Chiang et al, ACL 2013. "
                             "'basic': a basic generalization of CKY that matches rules according to an arbitrary "
                             "visit order on edges (less efficient).")
    parser.add_argument("-bn", "--boundary_nodes", action="store_true",
                        help="In the tree decomposition parser, use the full representation for graph fragments "
                             "instead of the compact boundary node representation. "
                             "This can provide some speedup for grammars with small rules.")
    parser.add_argument("-v", "--verbose", type=int, default=2,
                        help="Stderr output verbosity: "
                             "0 (all off), 1 (warnings), 2 (info, default), 3 (details), 3 (debug)")

    args = parser.parse_args()

    # Definition of logger output verbosity levels 
    log.LOG = {0:{log.err},
               1:{log.err, log.warn},
               2:{log.err, log.warn, log.info},
               3:{log.err, log.warn, log.info, log.chatter},
               4:{log.err, log.warn, log.chatter, log.info, log.debug}
              }[args.verbose]

    main(
        args.data_dir,
        args.config,
        args.grammar_file,
        args.parser,
        args.boundary_nodes,
    )
