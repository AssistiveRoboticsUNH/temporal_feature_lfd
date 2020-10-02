import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csv_output/feature_split.csv")
df["file"] = df["file"].str.split('/').str[-1]
df["label"] = df["file"].str.split('_').str[0]
print(df.columns)

df = df.groupby("file")
print(df.head(10))

#df.plot()
#plt.show()