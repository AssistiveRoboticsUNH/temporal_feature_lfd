import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("csv_output/feature_split.csv")
df["file"] = df["file"].str.split('/').str[-1]
df["label"] = df["file"].str.split('_').str[0]

print(df.columns)
df = df.drop(columns=[df.columns[0]])


print(df)

#df_s = df.groupby(["label"])
#print(df_s.head(10))

#df_s = pd.DataFrame({"label": df["label"], "file": df["file"], "feature": df["feature_1"]})
#df.boxplot()

fig, ax = plt.subplots(figsize=(10,10))

#sns.stripplot(ax=ax, data=df[df["label"] == "r"], jitter=True, color='r', alpha=.1)
#sns.stripplot(ax=ax, data=df[df["label"] == "b"], jitter=True, color='b', alpha=.1)
sns.boxplot(ax=ax, data=df, hue='label')
#sns.boxplot(ax=ax, data=df[df["label"] == "r"], color='r')
#sns.boxplot(ax=ax, data=df[df["label"] == "b"], color='b')

plt.xticks(rotation=90)
#df_s.plot.box(df["feature_1"], c=df["label"])
plt.tight_layout()
plt.show()