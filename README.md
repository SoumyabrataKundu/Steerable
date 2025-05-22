# Steerable

This package implements the Steerable convolutiona and transformer architectures in two and three dimensions. To use the package just clone the package in the working directory. To import the neural networks architectures, run


```{python}
import Steerable.nn as snn
```


## Two Dimensions

In two dimensions the convolution is used as

```{python}
layer1 = snn.SE2ConvType1(1, 2, 3, n_radius=n_radius,n_theta=n_theta, max_m=max_m, padding='same')
layer2 = snn.SE2ConvType1(2, 2, 3, n_radius=n_radius,n_theta=n_theta, max_m=max_m, padding='same')
```