def apply_threshold(probability, threshold=0.5):
    return 1 if probability>=threshold else 0