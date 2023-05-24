# population_distance
Calculate distance between two populations.

## Requirement
- Python >= 3.9   
- numpy

## Usage
Make the target class inherit from `base.Located` or `base.ValuedLocated` and assign it to the `distance_measure.measure()` function.
```python
import base
import distance_measure

class Point(base.Located):
    def __init__(self, x: float):
        self.x = x
    
    def get_location(self) -> float:
        return self.x


points1 = [Point(2.0)]
points2 = [Point(-0.3), Point(3.0)]

dist = distance_measure.measure(points1, points2)

print(dist)
# 1.0
```

## Mathematics
The distance is calculated based on the number of operations that match two sets. 
Specifically, the following three items are considered  
1. Difference in the number of elements  
1. Euclidean distance of corresponding elements  
1. Difference in the values of the corresponding elements (always 0 in the case of the `Located` class)

Since the axiom of distance is satisfied, the two populations are commutative.

```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = D(\mathbf{p}^{(2)}, \mathbf{p}^{(1)})
```
where $\mathbf{p}^{(1)}, \mathbf{p}^{(2)}$ are arrays representing populations.

### Case of `Located` class
Calculate the "distance" according to the following formula:

```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = w_o |N^{(1)} - N^{(2)}| + w_d \sum_{i=0}^{N^{(1)}} |p_{i}^{(1)} - p_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|}^{(2)}|_{L2}
```

where $N^{(1)}, N^{(2)}$ are the number of elements in $\mathbf{p}^{(1)}, \mathbf{p}^{(2)}$ respectively, satisfying $N^{(1)} \leq N^{(2)}$.
And $w_o, w_d$ are weight for the operation and Euclidean distance, respectively, satisfying

```math
w_o + w_d = 1\ .
```

Since ${\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|$ is an expression that returns the index $j$ of $\mathbf{p}^{(2)}$ closest to point $p_i^{(1)}$, 
the content of the $\sum$ of the second term represents the distance to the $p_j^{(2)}$ closest to $p_i^{(1)}$ in the sense of Euclidean distance.
### Case of `valuedLocated` class
In case of `valuedLocated` class, calculate the "distance" according to the following formula:

```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = w_o |N^{(1)} - N^{(2)}| + w_d \sum_{i=0}^{N^{(1)}} |p_{i}^{(1)} - p_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|  + |v_i^{(1)} - v_j^{(2)}|}^{(2)}|_{L2} +  w_v \sum_{i=0}^{N^{(1)}} |v_{i}^{(1)} - v_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|  + |v_i^{(1)} - v_j^{(2)}|}^{(2)}|
```

As for $N^{(1)}, N^{(2)}$ and $\bm{p}^{(1)}, \mathbf{p}^{(2)}$, it is the same as in the section above.
The difference from the above section is the $v^{(1)}, v^{(2)}$, which represents the value the element $\mathbf{p}^{(1)}, \mathbf{p}^{(2)}$ has, respectively.
And $w_o, w_d, w_v$ are weight for the operation, Euclidean distance and values, respectively, satisfying

```math
w_o + w_d + w_v = 1\ .
```

### Distance from empty population
If one population is empty, the following formula applies:

```math
D_\emptyset(\mathbf{p}) = w_o N + w_d \sum_{i=0}^{N} |p_i|_{L2} + w_v \sum_{i=0}^{N} |v_i|
```

### Example
The process of calculating the distance in [Usage](#Usage) described above is as follows:
```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = w_o |N^{(1)} - N^{(2)}| + w_d \sum_{i=0}^{N^{(1)}} |p_{i}^{(1)} - p_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|}^{(2)}|_{L2}
```
```math
= \frac{1}{2} |1-2| + \frac{1}{2} |p_{0}^{(1)} - p_{{\rm argmin}_j |p_0^{(1)} - p_j^{(2)}|}^{(2)}|_{L2} \\
```
```math
= \frac{1}{2} |1-2| + \frac{1}{2} |p_{0}^{(1)} - p_{{\rm argmin} (|2 - (-0.3)|,\ |2-3|)}^{(2)}| \\
```
```math
= \frac{1}{2} |1-2| + \frac{1}{2} |2 - 3| \\
```
```math
= 1
```
