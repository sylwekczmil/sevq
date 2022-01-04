======================
Performance Comparison
======================

We compared performance of the new classification algorithm and the state-of-the-art traditional algorithms AB, DT, GNB, KNN, MLP, NC, QDA, RF, SVC and XGB and incremental algorithms SFAM, HT, OB, HAT, KNNI, NB, EFDT, ARF, AEE, DWM, LVQ, LVQ2, LVQ2.1, LVQ3 and EVQ. Results are presented in separate tables for traditional and incremental algorithms because the SFAM method and algorithms from the LVQ family -- with which we primarily wanted to compare the results -- require normalized data. Hence, we used normalized data for testing incremental algorithms, and non-normalized data for testing traditional ones.

The average results obtained for the 36 datasets presented for each algorithm were calculated from each of the fold results on each dataset.

To date, no common consensus has emerged as to the choice of measures used to evaluate the performance of classifiers for comparison of data classification algorithms. Thus, in this study, we decided to choose the most popular measures, such as accuracy, precision, sensitivity, F1 score and AUC, to evaluate and compare classifiers.

Metrics
--------------------------------------
Table 1 and Table 2 present the average results obtained for the 36 datasets for each algorithm, calculated from each of the fold results on each dataset. Results in the tables are sorted by AUC in the descending order.

.. csv-table::
   :file: ./data/comparison_result_rounded.csv
   :header-rows: 1

Table 1. Results of a comparison of incremental algorithms.

.. csv-table::
   :file: ./data/comparison_normalized_result_rounded.csv
   :header-rows: 1

Table 2. Results of a comparison of incremental algorithms.

Given such an ordering among the traditional algorithms, XGB is the clear
winner because it achieved the best results for each of the metrics.
Additionally, SEVQ is the third-best for AUC and the fifth-best for
accuracy. Among incremental algorithms, SEVQ achieved the best results for
AUC, and was second-best for accuracy.

Ranking of the compared algorithms
--------------------------------------
For each dataset, the means of AUC and accuracy for the 10 folds were
calculated both for traditional and incremental algorithms. The algorithm
with the highest average value was ranked first. The counts of wins and
the instances of placing second and third are presented in Tables 3 and 4.

.. csv-table::
   :file: ./data/comparison.csv
   :header-rows: 1

Table 3. Ranking of compared traditional algorithms.

Among the traditional algorithms, XGB won most often. SEVQ took the third
place in this ranking.

.. csv-table::
   :file: ./data/comparison_normalized.csv
   :header-rows: 1

Table 4. Ranking of compared incremental algorithms.

Among the incremental algorithms, SEVQ scored the highest number of wins
due to the highest AUC on 6 datasets and due to the highest accuracy on 5
datasets among incremental algorithms, surpassing even SFAM, LVQ, LVQ2,
LVQ2.1 and LVQ3.

Means of accuracy and AUC
--------------------------------------
We also compared the results of the traditional algorithms and SEVQ
using box plots. A box plot is a standardized type of chart often used
in explanatory data analysis. Such a plot can be used to visualize the
distribution of numerical data and skewness by displaying the data
quartiles and averages. A box plot shows the summary of a set of data,
including the minimum score, the lower quartile, the median, the upper
quartile and the maximum score. An outlier in a box plot is an observation
that is numerically distant from the rest of the data. When a box plot is
analyzed, an outlier is defined as a data point located outside the
whiskers of the box plot.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/comparison_plot/comparison_accuracy.png
   :width: 800

Figure 1. Distribution of ACC values for each traditional algorithm across all datasets.

Figure 1 presents a box plot of accuracy for each traditional algorithm
across 36 datasets subjected to 10-fold cross-validation. The box plots
are arranged in the descending order of medians of accuracy. SEVQ is in
the second position among all traditional algorithms tested for accuracy;
in other words, it is a good algorithm for general usage. It only loses
to the XGB algorithm, which has become the best non-incremental algorithm
for winning competitions at Kaggle because it is extremely powerful.
The plot also shows several outliers that lower the average results of
the algorithm and its position in Table 1 and 3.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/comparison_plot/comparison_auc.png
   :width: 800

Figure 2. Distribution of AUC values for each traditional algorithm across
all datasets.

Figure 2 shows a box plot of AUC for each of the traditional algorithms
arranged in the descending order of medians of AUC. SEVQ, as before, is
in the second position, and again only loses to the XGB algorithm.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/comparison_plot/comparison_normalized_accuracy.png
   :width: 800

Figure 3. Distribution of ACC values for each incremental algorithm across
all datasets.

Figure 3 presents a box plot of accuracy arranged in the descending order
of medians of accuracy. SEVQ is in the first position among all incremental
algorithms tested for accuracy. The plot also shows several outliers that
lower the average results of the algorithm and its position in Tables 2
and 4.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/comparison_plot/comparison_normalized_auc.png
   :width: 800

Figure 4. Distribution of AUC values for each incremental algorithm across
all datasets.

Figure 4 shows a box plot of AUC for each of the incremental algorithms
arranged in the descending order of medians of AUC. SEVQ, as before, is
in the first position.


