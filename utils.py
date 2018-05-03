def average_precision_at_k(k, class_correct):
    # return average precision at k.
    # more examples: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    # and: https://www.kaggle.com/c/avito-prohibited-content#evaluation
    # class_correct is a list with the binary correct label ordered by confidence level.
    assert k <= len(class_correct) and k > 0
    score = 0.0
    hits = 0.0
    for i in range(k):
        if class_correct[i] == 1:
            hits += 1.0
        score += hits / (i + 1.0)
    score /= k
    return score
