=======================
Algorithm Visualisation
=======================

The 2D visualizations shown in Figures 1 - 4 were prepared to demonstrate the operation of SEVQ. Each figure shows a plot of records and categories generated for records on the left, and the corresponding accuracy plot for the number of records processed by the algorithm on the right to emphasize the incremental nature of this algorithm. Figures have been prepared only for visualization purposes, and in these cases, learning and testing were performed on the same data, and the algorithm without epochs was used.

To start our test of the SEVQ algorithm, we randomly generated a multiclass dataset that contained 100 data points. Normally distributed clusters of points were assigned to each class. The dataset is presented in Figure 1 on the left. Individual samples are marked on the plot with green, blue and red triangles.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/3_blobs.png
   :width: 49%
.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/3_blobs-incremental.png
   :width: 49%

Figure 1. Blobs artificial dataset: (a) data visualization before and after the learning process, (b) accuracy vs. the number of learning samples.


We tested SEVQ on this synthetic dataset to resolve the problem of classifying samples into one of three classes and observe how many categories would be created by the algorithm. The latter generated exactly one category marked with a filled circle for each class. The line plot for accuracy on the right shows that the training process converged well, and the SEVQ algorithm required only five examples to reach 100% accuracy. In the case shown in Figure 1, one example was randomly selected from the 0th class, three examples were from the 1st class and the last example was from the 2nd class. In the optimistic case, if one case were randomly selected from each class, it would certainly turn out that only three examples would suffice to obtain 100% accuracy.

Continuing our test of SEVQ, we generated a new synthetic dataset with 100 data points arranged as a pair of moons facing each other in an asymmetrical arrangement, as shown in Figure 2. These moons are not linearly separable.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/moons.png
   :width: 49%
.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/moons-incremental.png
   :width: 49%

Figure 2. Moons artificial dataset: (a) data visualization before and after the learning process, (b) accuracy vs. the number of learning samples.

SEVQ allocated the categories very well. For samples from the first category, marked with red triangles, it generated three categories, and for samples from the second, four categories were created. This experimental scenario was more complicated than the previous one; therefore, the algorithm needs more examples. A validation accuracy of 100% was achieved after providing 31 randomly selected samples.

Another type of natural patterns is concentric circles. For the test, we generated two concentric circles with 50 data points in each, which were assigned to the two respective classes shown in Figure 3. SEVQ created 16 classes for red samples and 14 for green ones. We observed an accuracy rate of 96% after providing all examples.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/circles.png
   :width: 49%
.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/circles-incremental.png
   :width: 49%

Figure 3. Circles artificial dataset: (a) data visualization before and after the learning process, (b) accuracy vs. the number of learning samples.

To further test our neural network model, we used a two-spiral dataset with 100 data points in each spiral, as shown in Figure 4. The two-spiral problem has become a standard benchmark for neural network algorithms since it was first proposed by Wieland. Although this task is easy to visualize, it is very difficult for many networks to learn such an arrangement of points due to its extreme nonlinearity.

.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/spirals.png
   :width: 49%
.. image:: https://raw.githubusercontent.com/sylwekczmil/sevq/main/data/research/generated/spirals-incremental.png
   :width: 49%

Figure 4. Two Spirals artificial dataset: (a) data visualization before and after the learning process, (b) accuracy vs. the number of learning samples for this artificial dataset.

SEVQ created 8 categories for each class. We were able to obtain 100% accuracy after providing 158 examples.

As shown in Figures 1 - 4, the order in which the examples are supplied affects the outcome; i.e., providing the samples in a different order would result in the generation of different categories. Experiments show that for the best results, records should be provided from different classes without class repetition. It is possible to improve the algorithm's results using multiple learning epochs; however, it is recommended to shuffle the records in subsequent epochs.

