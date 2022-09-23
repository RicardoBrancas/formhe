def hamming(str_a, str_b):
    if len(str_a) != len(str_b):
        raise ValueError()

    dist = 0
    for a, b in zip(str_a, str_b):
        dist += 1 if a != b else 0

    return dist / len(str_a)
