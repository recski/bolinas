from partial.oie import get_labels


def f1(prec, rec):
    try:
        return 2 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        return 0


def filter_for_pr(derivations, gold_labels, metric):
    derivations_to_keep = [None] * len(gold_labels)
    max_scores = [-1.0] * len(gold_labels)
    for score, derivation in derivations:
        labels = get_labels(derivation)
        labels = {k: v for k, v in labels.items() if v != "X"}
        for i, gold in enumerate(gold_labels):
            gold_nodes = set(gold.keys())
            derived_nodes = set(labels.keys())
            if metric == "prec":
                penalty = len(derived_nodes) / float(len(gold_nodes))
                if penalty > 1.0:
                    penalty = 1.0
                score = (len(gold_nodes & derived_nodes) / float(len(derived_nodes))) * penalty
            elif metric == "rec":
                score = len(gold_nodes & derived_nodes) / float(len(gold_nodes))
            else:
                assert metric == "f1"
                prec = len(gold_nodes & derived_nodes) / float(len(derived_nodes))
                rec = len(gold_nodes & derived_nodes) / float(len(gold_nodes))
                score = f1(prec, rec)
            if score > max_scores[i]:
                derivations_to_keep[i] = derivation
                max_scores[i] = score
    assert None not in derivations_to_keep
    ret = zip(max_scores, derivations_to_keep)
    ret = sorted(ret, reverse=True, key=lambda x: x[0])
    return ret
