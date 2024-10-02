def resolve_pred(pred_labels, pos_tags, pp, top_order):
    preds = [n for n, l in pred_labels.items() if l == "P"]
    if pp == "keep" and len(preds) > 0:
        return
    verbs = [n for n, t in pos_tags.items() if t == "VERB"]
    preds_w_verbs = [n for n in preds if n in verbs]
    if len(preds) == 1:
        assert pp != "keep"
        if preds == preds_w_verbs:
            return
        else:
            assert len(preds_w_verbs) == 0
    if len(preds_w_verbs) == 1:
        keep = preds_w_verbs[0]
        for p in preds:
            if p != keep:
                del pred_labels[p]
        return
    if len(preds_w_verbs) == 0:
        for p in preds:
            del pred_labels[p]
    if len(verbs) == 0:
        pred_labels[top_order[1]] = "P"
        return
    if len(verbs) == 1:
        pred_labels[verbs[0]] = "P"
        return
    assert len(verbs) > 1
    if len(preds_w_verbs) > 1:
        verbs = preds_w_verbs
    first_verb_idx = None
    for v_idx in verbs:
        idx = top_order.index(int(v_idx))
        if first_verb_idx is None or idx < first_verb_idx:
            first_verb_idx = idx
    first_verb_node = str(top_order[first_verb_idx])
    pred_labels[first_verb_node] = "P"
    if len(preds_w_verbs) > 1:
        for p in preds:
            if p != first_verb_node:
                del pred_labels[p]
    return


def add_arg_idx(extracted_labels, length):
    prev = "O"
    idx = -1
    for i in range(1, length + 1):
        if str(i) not in extracted_labels:
            extracted_labels[str(i)] = "O"
        else:
            if extracted_labels[str(i)] == "A":
                if not prev.startswith("A"):
                    idx += 1
                extracted_labels[str(i)] = "A" + str(idx)
        prev = extracted_labels[str(i)]


def postprocess(extracted_labels, pos_tags, top_order, pp):
    resolve_pred(extracted_labels, pos_tags, pp, top_order)
    add_arg_idx(extracted_labels, len(pos_tags))

