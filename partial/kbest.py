#!/usr/bin/env python2
import datetime
import json
import math
import os.path
import pickle
import time

from argparse import ArgumentParser
from copy import copy

from common import log
from common import output
from common.exceptions import DerivationException
from partial.filter.pr_filter import filter_for_pr
from partial.filter.preproc import get_gold_labels, get_pos_tags
from partial.filter.size_filter import filter_for_size
from partial.oie import get_labels, get_rules
from partial.utils import get_range


def get_k_best_unique_derivation(chart, k):
    kbest_unique_nodes = set()
    kbest_unique_derivations = []
    for score, derivation in chart:
        final_item = derivation[1]["START"][0]
        nodes = sorted(list(final_item.nodeset), key=lambda node: int(node[1:]))
        nodes_str = " ".join(nodes)
        if nodes_str not in kbest_unique_nodes:
            kbest_unique_nodes.add(nodes_str)
            kbest_unique_derivations.append((score, derivation))
        if len(kbest_unique_derivations) >= k:
            break
    assert len(kbest_unique_derivations) == len(kbest_unique_nodes)
    if len(kbest_unique_derivations) < k:
        log.info("Found only %i derivations." % len(kbest_unique_derivations))
    return kbest_unique_derivations


def extract_for_kth_derivation(derivation, n_score, matches_lines, labels_lines, rules_lines, sen_log_lines, ki):
    shifted_derivation = output.print_shifted(derivation)
    matches_lines.append("%s;%g\n" % (shifted_derivation, n_score))

    labels = get_labels(derivation)
    labels_lines.append("%s\n" % json.dumps(labels))

    format_derivation = output.format_derivation(derivation)
    rules_lines.append("%s\t#%g\n" % (format_derivation, n_score))
    rules = get_rules(derivation)
    for grammar_nr, rule_str in sorted(rules.items()):
        prob = rule_str.split(';')[1].strip()
        rule = rule_str.split(';')[0].strip()
        rules_lines.append("%s\t%.2f\t%s\n" % (grammar_nr, float(prob), rule))
    rules_lines.append("\n")

    final_item = derivation[1]["START"][0]
    nodes = sorted(list(final_item.nodeset), key=lambda node: int(node[1:]))
    sen_log_lines.append("\nk%d:\t%s" % (ki, nodes))


def save_output(outputs):
    for (fn, lines) in outputs:
        with open(fn, "w") as f:
            f.writelines(lines)


def main(data_dir, config_file):
    start_time = time.time()
    logprob = True
    config = json.load(open(config_file))

    log_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "log",
        "kbest_" + config["model_dir"] + ".log"
    )
    log_lines = [
        "Execution start: %s\n" % str(datetime.datetime.now()),
        "Chart filters: %s" % " ".join([f for f, c in config["filters"].items() if not c.get("ignore", False)]),
        "\n"]
    first = config.get("first", None)
    last = config.get("last", None)
    if first:
        log_lines.append("First: %d\n" % first)
    if last:
        log_lines.append("Last: %d\n" % last)

    score_disorder_collector = {}
    model_dir = os.path.join(data_dir, config["model_dir"])
    for sen_idx in get_range(model_dir, first, last):
        print "\nProcessing sen %d\n" % sen_idx
        sen_dir_out = os.path.join(model_dir, str(sen_idx))

        bolinas_dir = os.path.join(sen_dir_out, "bolinas")
        chart_file = os.path.join(bolinas_dir, "sen" + str(sen_idx) + "_chart.pickle")
        if not os.path.exists(chart_file):
            continue

        with open(chart_file) as f:
            chart = pickle.load(f)

        if "START" not in chart:
            print "No derivation found"
            continue

        gold_labels = get_gold_labels(os.path.join(data_dir, config["preproc_dir"]), sen_idx)
        top_order = json.load(open(os.path.join(
            data_dir,
            config["preproc_dir"],
            str(sen_idx),
            "preproc",
            "pos_edge_graph_top_order.json"
        )))
        pos_tags = get_pos_tags(os.path.join(
            data_dir,
            config["preproc_dir"],
            str(sen_idx),
            "preproc",
            "parsed.conll"
        ))
        for name, c in config["filters"].items():
            if c.get("ignore", False):
                continue
            print "Processing " + name
            matches_lines = []
            labels_lines = []
            rules_lines = []
            sen_log_lines = []

            filtered_chart = copy(chart)
            sen_log_lines.append("Chart 'START' length: %d\n" % len(filtered_chart["START"]))
            if "chart_filter" in c:
                chart_filter = c["chart_filter"]
                assert chart_filter in ["basic", "max"]
                filtered_chart = filter_for_size(chart, chart_filter)
            sen_log_lines.append("Chart 'START' length after size filter: %d\n" % len(filtered_chart["START"]))

            derivations = filtered_chart.derivations("START")

            assert ("k" in c and "pr_metric" not in c) or ("k" not in c and "pr_metric" in c)

            if "k" in c:
                k_best_unique_derivations = get_k_best_unique_derivation(derivations, c["k"])
            elif "pr_metric" in c:
                metric = c["pr_metric"]
                assert metric in ["prec", "rec", "f1"]
                k_best_unique_derivations = filter_for_pr(derivations, gold_labels, metric, pos_tags, top_order)
            else:
                print "Neither 'k' nor 'pr_metric' is set"
                continue

            last_score = None
            score_disorder = {}
            for i, (score, derivation) in enumerate(k_best_unique_derivations):
                ki = i + 1
                if "k" in c:
                    n_score = score if logprob else math.exp(score)
                else:
                    n_score = score

                new_score = score
                if last_score:
                    if new_score > last_score:
                        order_str = "%d-%d" % (ki - 1, ki)
                        score_disorder[order_str] = (last_score, new_score)
                last_score = new_score

                try:
                    extract_for_kth_derivation(
                        derivation,
                        n_score,
                        matches_lines,
                        labels_lines,
                        rules_lines,
                        sen_log_lines,
                        ki,
                    )
                except DerivationException, e:
                    log.err("Could not construct derivation: '%s'. Skipping." % e.message)

            for i, val in score_disorder.items():
                sen_log_lines.append("%s: %g / %g\n" % (i, val[0], val[1]))
            score_disorder_collector[sen_idx] = (len(score_disorder.items()), len(k_best_unique_derivations))

            out_dir = os.path.join(bolinas_dir, name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_output(
                [
                    (os.path.join(out_dir, "sen" + str(sen_idx) + "_matches.graph"), matches_lines),
                    (os.path.join(out_dir, "sen" + str(sen_idx) + "_predicted_labels.txt"), labels_lines),
                    (os.path.join(out_dir, "sen" + str(sen_idx) + "_derivation.txt"), rules_lines),
                    (os.path.join(out_dir, "sen" + str(sen_idx) + ".log"), sen_log_lines),
                ]
            )

    elapsed_time = time.time() - start_time
    log_lines.append("Execution finish: %s\n" % str(datetime.datetime.now()))
    time_str = "Elapsed time: %d min %d sec" % (elapsed_time / 60, elapsed_time % 60)
    print time_str
    log_lines.append(time_str)
    log_lines.append("\n")
    num_sem = len(score_disorder_collector.keys())
    sum_score_disorder = sum([val[0] for val in score_disorder_collector.values()])
    log_lines.append("Number of sentences: %d\n" % num_sem)
    log_lines.append("Sum of score disorders: %d\n" % sum_score_disorder)
    avg_str = "Average score disorders: %.2f\n" % (sum_score_disorder / float(num_sem))
    log_lines.append(avg_str)
    log_lines.append("\n")
    print avg_str
    with open(log_file, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description ="Bolinas is a toolkit for synchronous hyperedge replacement grammars.")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument('-c', '--config-file')
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
        args.config_file,
    )
