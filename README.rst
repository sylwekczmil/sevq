====
sevq
====


.. image:: https://img.shields.io/pypi/v/sevq.svg
        :target: https://pypi.python.org/pypi/sevq

.. image:: https://img.shields.io/travis/sylwekczmil/sevq.svg
        :target: https://travis-ci.com/github/sylwekczmil/sevq

.. image:: https://readthedocs.org/projects/sevq/badge/?version=latest
        :target: https://sevq.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


SEVQ: Simple Evolving Vector Quantization


* Free software: MIT license
* Documentation: https://sevq.readthedocs.io.



Installation
--------------

To install sevq, run this command in your terminal:

.. code-block:: console

    pip install sevq

Usage
-----

Training and prediction one sample at a time


.. code:: python3

    from sevq.algorithm import SEVQ

    c = SEVQ()
    c.partial_fit([-2, -2], 2)
    c.partial_fit([-1, -1], 1)
    c.partial_fit([1, 1], 1)
    c.partial_fit([2, 2], 2)

    print(c.predict([0, 0]))  # 1
    print(c.predict([3, 3]))  # 2
    print(c.predict([-3, -3]))  # 2

Training and prediction on multiple samples


.. code:: python3

    from sevq.algorithm import SEVQ

    c = SEVQ()
    c.fit(
        [[-2, -2], [-1, -1], [1, 1], [2, 2]],
        [2, 1, 1, 2],
        epochs=1, permute=False
    )

    print(c.predict([[0, 0], [3, 3], [-3, -3]]))  # [1, 2, 2]
