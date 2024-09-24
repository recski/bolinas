def get_labels(derivation):
    if type(derivation) is not tuple:
        if derivation == "START" or derivation.rule.symbol == "S":
            return {}
        return {derivation.mapping['_1'].split('n')[1]: derivation.rule.symbol}
    else:
        ret = {}
        items = [c for (_, c) in derivation[1].items()] + [derivation[0]]
        for item in items:
            for (k, v) in get_labels(item).items():
                assert k not in ret
                ret[k] = v
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
