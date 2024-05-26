---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="bhWV8oes-wKR" -->
# COURSE: A deep understanding of deep learning
## SECTION: ANNs
### LECTURE: Model depth vs. breadth
#### TEACHER: Mike X Cohen, sincxpress.com
##### COURSE URL: udemy.com/course/deeplearning_x/?couponCode=202401
<!-- #endregion -->

```python id="YeuAheYyhdZw"
# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
```

<!-- #region id="1ViJutqaaNb2" -->
# Import and organize the data
<!-- #endregion -->

```python id="MU7rvmWuhjud"
# import dataset
import pandas as pd
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# convert from pandas dataframe to tensor
data = torch.tensor( iris[iris.columns[0:4]].values ).float()

# transform species to number
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species=='setosa'] = 0 # don't need!
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2
```

<!-- #region id="jCuMSE6baRar" -->
# Construct and sanity-check the model
<!-- #endregion -->

```python id="eZMzMLxfULjf"
# create a class for the model

class ANNiris(nn.Module):
  def __init__(self,nUnits,nLayers):
    super().__init__()

    # create dictionary to store the layers
    self.layers = nn.ModuleDict()
    self.nLayers = nUnits#nLayers#

    ### input layer
    self.layers['input'] = nn.Linear(4,nUnits)

    ### hidden layers
    for i in range(nLayers):
      self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)

    ### output layer
    self.layers['output'] = nn.Linear(nUnits,3)


  # forward pass
  def forward(self,x):
    # input layer (note: the code in the video omits the relu after this layer)
    x = F.relu( self.layers['input'](x) )

    # hidden layers
    for i in range(self.nLayers):
      x = F.relu( self.layers[f'hidden{i}'](x) )

    # return output layer
    x = self.layers['output'](x)
    return x
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1676840556776, "user": {"displayName": "Mike X Cohen", "userId": "13901636194183843661"}, "user_tz": -540} id="2-GmvNgEYgHK" outputId="64aaf42f-9ea9-4c3f-d621-40ede6bdc0d9"
# generate an instance of the model and inspect it
nUnitsPerLayer = 12
nLayers = 4
net = ANNiris(nUnitsPerLayer,nLayers)
net
```

```python colab={"base_uri": "https://localhost:8080/", "height": 381} executionInfo={"elapsed": 878, "status": "error", "timestamp": 1676840557650, "user": {"displayName": "Mike X Cohen", "userId": "13901636194183843661"}, "user_tz": -540} id="XwtrXLSNYyC8" outputId="61867ff5-2e94-4f91-ec8c-cde7008638fd"
# A quick test of running some numbers through the model.
# This simply ensures that the architecture is internally consistent.


# 10 samples, 4 dimensions
tmpx = torch.randn(10,4)

# run it through the DL
y = net(tmpx)

# exam the shape of the output
print( y.shape ), print(' ')

# and the output itself
print(y)
```

<!-- #region id="YL7cvyjUaXjc" -->
# Create a function that trains the model
<!-- #endregion -->

```python id="cVD1nFTli7TO"
# a function to train the model

def trainTheModel(theModel):

  # define the loss function and optimizer
  lossfun = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(theModel.parameters(),lr=.01)

  # loop over epochs
  for epochi in range(numepochs):

    # forward pass
    yHat = theModel(data)

    # compute loss
    loss = lossfun(yHat,labels)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



  # final forward pass to get accuracy
  predictions = theModel(data)
  predlabels = torch.argmax(predictions, axis=1)
  acc = 100*torch.mean((predlabels == labels).float())

  # total number of trainable parameters in the model
  nParams = sum(p.numel() for p in theModel.parameters() if p.requires_grad)

  # function outputs
  return acc,nParams
```

```python id="41R4X0MCaxVc"
# test the function once

C
```python id="JYouZAY4i3jM"
# show accuracy as a function of model depth
fig, ax = plt.subplots(1,figsize=(12,6))

ax.plot(numunits,accuracies,'o-',markerfacecolor='w',markersize=9)
ax.plot(numunits[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(numunits[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.legend(numlayers)
ax.set_ylabel('accuracy')
ax.set_xlabel('Number of hidden units')
ax.set_title('Accuracy')
plt.show()
```

```python id="St6NI4qBk4tO"
# Maybe it's simply a matter of more parameters -> better performance?

# vectorize for convenience
x = totalparams.flatten()
y = accuracies.flatten()

# correlation between them
r = np.corrcoef(x,y)[0,1]

# scatter plot
plt.plot(x,y,'o')
plt.xlabel('Number of parameters')
plt.ylabel('Accuracy')
plt.title('Correlation: r=' + str(np.round(r,3)))
plt.show()
```

```python id="3ix4I-SgJzXX"

```

<!-- #region id="JmraVzTcJ0x1" -->
# Additional explorations
<!-- #endregion -->

```python id="pml6nCTcAMWC"
# 1) Try it again with 1000 training epochs. Do the deeper models eventually learn?
#
# 2) The categories are coded a "0", "1", and "2". Is there something special about those numbers?
#    Recode the labels to be, e.g., 5, 10, and 17. Or perhaps -2, 0, and 2. Is the model still able to learn?
#
```
