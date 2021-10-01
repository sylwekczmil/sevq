from research.comparison import process_comparison, process_comparison_results, process_comparison_result_winners, \
    process_comparison_results_charts
from research.generated import process_generated
from research.helpers.data import download_data
from research.helpers.util import seed_everything
from research.info import dataset_info, algorithm_info
from research.wilcoxon import process_wilcoxon

if __name__ == '__main__':
    seed_everything()
    download_data()
    dataset_info()
    algorithm_info()
    process_generated()

    for normalized in [False, True]:
        process_comparison(normalized)
        process_comparison_results(normalized)
        process_comparison_results_charts(normalized)
        process_comparison_result_winners(normalized)
        process_wilcoxon(normalized)

        try:
            # works only if Python rpy2==3.4.3, and R install.packages("ScottKnottESD") packages installed
            from research.scott_knott import process_scott_knott

            process_scott_knott(normalized=normalized)
        except:
            pass
