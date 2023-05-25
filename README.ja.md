# population_distance
二つの集団間の距離を計算します。

## 要件
- Python >= 3.9   
- numpy

## 使い方
対象となるクラスに `base.Located` か `base.ValuedLocated` を継承させ、`distance_measure.measure()` 関数に代入します。

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

## 数理
集団間の距離は、二つの集団を一致させる操作の回数に基づいて計算されます。
具体的には以下の3要素を考慮します。  
1. 要素数の差(多い方の集団ぁら要素を削除する回数)  
1. 対応する要素のユークリッド距離  
1. 対応する要素の値の差 (`Located` クラスの場合、常に 0 になります)

距離の公理を満たすので、二つの集団は可換です。

```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = D(\mathbf{p}^{(2)}, \mathbf{p}^{(1)})
```
ここで $\mathbf{p}^{(1)}, \mathbf{p}^{(2)}$ 集団要素の配列を表します。

### `Located` クラスの場合
集団間の距離は以下の式に従って計算されます:

```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = w_o |N^{(1)} - N^{(2)}| + w_d \sum_{i=0}^{N^{(1)}} |p_{i}^{(1)} - p_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|}^{(2)}|_{L2}
```

ここで $N^{(1)}, N^{(2)}$ はそれぞれ $\mathbf{p}^{(1)}, \mathbf{p}^{(2)}$ の要素数で、$N^{(1)} \leq N^{(2)}$ を満たします。
また $w_o, w_d$ はそれぞれ要素を削除する操作とユークリッド距離にかかる重みを表し、
```math
w_o + w_d = 1\ .
```
を満たします。

${\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|$ は $p_i^{(1)}$ に最も近い $\mathbf{p}^{(2)}$ のインデックス $j$ を返すので、
第2項の $\sum$ の中身は $p_i^{(1)}$ に最も近い $p_j^{(2)}$ とのユークリッド距離を表します。
### `valuedLocated` クラスの場合
`valuedLocated` クラスの場合、距離は以下の式に従って計算されます:

```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = w_o |N^{(1)} - N^{(2)}| + w_d \sum_{i=0}^{N^{(1)}} |p_{i}^{(1)} - p_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|  + |v_i^{(1)} - v_j^{(2)}|}^{(2)}|_{L2} +  w_v \sum_{i=0}^{N^{(1)}} |v_{i}^{(1)} - v_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|  + |v_i^{(1)} - v_j^{(2)}|}^{(2)}|
```

$N^{(1)}, N^{(2)}$ 及び $\mathbf{p}^{(1)}, \mathbf{p}^{(2)}$ は上のセクションと同じ意味です。
上のセクションとの差は $v^{(1)}, v^{(2)}$ で、これはそれぞれ $\mathbf{p}^{(1)}, \mathbf{p}^{(2)}$ がの要素が持つ値を表します。
また、$w_o, w_d, w_v$ はそれぞれ削除操作、ユークリッド距離、値にかかる重みを表し、
```math
w_o + w_d + w_v = 1\ .
```
を満たします。

### 空の集団との距離
もし、片方の集団が空の場合は以下の式が適用されます。

```math
D_\emptyset(\mathbf{p}) = w_o N + w_d \sum_{i=0}^{N} |p_i|_{L2} + w_v \sum_{i=0}^{N} |v_i|
```

### 例
上述した [使い方](#使い方) での距離計算の過程は以下の通りです:
```math
D(\mathbf{p}^{(1)}, \mathbf{p}^{(2)}) = w_o |N^{(1)} - N^{(2)}| + w_d \sum_{i=0}^{N^{(1)}} |p_{i}^{(1)} - p_{{\rm argmin}_j |p_i^{(1)} - p_j^{(2)}|}^{(2)}|_{L2}
```
```math
= \frac{1}{2} |1-2| + \frac{1}{2} |p_{0}^{(1)} - p_{{\rm argmin}_j |p_0^{(1)} - p_j^{(2)}|}^{(2)}|_{L2}
```
```math
= \frac{1}{2} |1-2| + \frac{1}{2} |p_{0}^{(1)} - p_{{\rm argmin} (|2 - (-0.3)|,\ |2-3|)}^{(2)}|
```
```math
= \frac{1}{2} |1-2| + \frac{1}{2} |2 - 3|
```
```math
= 1
```
