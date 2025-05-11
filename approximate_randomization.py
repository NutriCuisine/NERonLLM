import numpy as np


# Return the likelihood that sample1's mean is greater than sample2's merely by chance
def chanceByChance(sample1, sample2, comparer=None, pairwise=True, repetitions=1000):
    if not comparer:
        comparer = lambda x, y: np.mean(x) - np.mean(y)
    true_diff = comparer(sample1, sample2)

    n = len(sample1)
    m = len(sample2)

    if pairwise and n != m:
        raise Exception("samples must be same size for pairwise. Got sample sizes {} and {}".format(n, m))

    combined = np.concatenate([sample1, sample2])

    def run_test(_):
        np.random.shuffle(combined)
        diff = comparer(combined[:n], combined[n:])
        return diff > true_diff

    def run_pairwise_test(_):
        swapper = np.random.rand(n) < 0.5

        # Ensure both samples have the same shape
        if sample1.shape != sample2.shape:
            raise ValueError("sample1 and sample2 must have the same shape for pairwise comparison.")

        s1new = np.where(swapper[:, None], sample1, sample2)
        s2new = np.where(swapper[:, None], sample2, sample1)

        diff = comparer(s1new, s2new)
        return diff >= true_diff

    test = run_pairwise_test if pairwise else run_test

    results = map(test, range(repetitions))

    return (sum(results) + 1) / (repetitions + 1)


def chanceByChanceDataFrame(dataframe, split_column, compare_column, left_value, right_value, comparer=None,
                            repetitions=1000):
    subsets = {}

    for category in dataframe[split_column].unique():
        subsets[category] = dataframe[dataframe[split_column] == category][compare_column].values

    return chanceByChance(subsets[left_value], subsets[right_value], comparer, pairwise=False, repetitions=repetitions)
