import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
# just data
df = pd.DataFrame(digits.data)

df['digits'] = digits.target
print(df)

print("-----------------------------------------------------------------")
print("-----------------------------------------------------------------")
print("-----------------------------------------------------------------")
images = digits.images
print(images[2])
print("##########################")
print(images[1])
print("###########################")

_, axes = plt.subplots(1, 13)
#images_and_labels = list((digits.images, digits.target))
for i in range(10):
  numbers = digits.target
  axes[i].set_title('IMGAGE: %i' % numbers[i])
  axes[i].imshow(images[i] ,cmap=plt.cm.gray_r, interpolation='nearest')
  #axes[i].imshow(images[i] ,cmap=plt.cm.gray, interpolation='nearest')

plt.show()