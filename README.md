# SEVQ: Simplified Evolving Vector Quantization

### To use this algorithm:

##### Installation

```bash
# create venv and activate..
# install algorithm
pip3 install git+https://github.com/sylwekczmil/sevq
```

##### Example usage

###### Training and prediction one sample at a time

```python3
from sevq.algorithm import SEVQ

c = SEVQ()
c.partial_fit([-2, -2], 2)
c.partial_fit([-1, -1], 1)
c.partial_fit([1, 1], 1)
c.partial_fit([2, 2], 2)

print(c.predict([0, 0]))  # 1 
print(c.predict([3, 3]))  # 2 
print(c.predict([-3, -3]))  # 2
```

###### Training and prediction on multiple samples

```python3
from sevq.algorithm import SEVQ

c = SEVQ()
c.fit(
    [[-2, -2], [-1, -1], [1, 1], [2, 2]],
    [2, 1, 1, 2],
    epochs=1, permute=False
)

print(c.predict([[0, 0], [3, 3], [-3, -3]]))  # [1, 2, 2]
```

### To replicate the research run:

```bash
# python 3.6 is required by neupu
git clone https://github.com/sylwekczmil/sevq
cd sevq
# create venv and activate
# up to you

# install dependencies
pip3 install -r requirements.txt

# run research code, please ignore warnings, this script can run long time
python3 run_research.py
```