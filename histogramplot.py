import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("train.csv")

import matplotlib.pyplot as plt

df.hist(figsize=(15, 10), bins=50, edgecolor='black')
plt.tight_layout()
plt.show()


