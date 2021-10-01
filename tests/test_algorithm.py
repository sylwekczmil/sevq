from sevq.algorithm import SEVQ


def test_one_by_one():
    c = SEVQ()
    c.partial_fit([-2, -2], 2)
    c.partial_fit([-1, -1], 1)
    c.partial_fit([1, 1], 1)
    c.partial_fit([2, 2], 2)

    assert c.predict([0, 0]) == 1
    assert c.predict([3, 3]) == 2
    assert c.predict([-3, -3]) == 2


def test_multi():
    c = SEVQ()
    c.fit(
        [[-2, -2], [-1, -1], [1, 1], [2, 2]],
        [2, 1, 1, 2],
        epochs=1, permute=False
    )

    assert all(c.predict([[0, 0], [3, 3], [-3, -3]]) == [1, 2, 2])
