import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("csv_output/feature_split.csv")
df["file"] = df["file"].str.split('/').str[-1]
df["label"] = df["file"].str.split('_').str[0]

df["diff"] = df["values"]
for f in df["feature"].unique():
    print(f)
    locs = df["feature"] == f

    min_v = df[locs]["values"].min()
    max_v = df[locs]["values"].max()

    #print(min_v, max_v)

    #df.loc[locs]["diff"] -= min_v
    df.loc[locs, "diff"] -= min_v

#print(df.columns)
#df = df.drop(columns=[df.columns[0]])

print(df)
df = df[df["diff"] > 0]
#df_s = df.groupby(["label"])
#print(df_s.head(10))

#df_s = pd.DataFrame({"label": df["label"], "file": df["file"], "feature": df["feature_1"]})
#df.boxplot()

fig, ax = plt.subplots(figsize=(10, 10))

#sns.stripplot(ax=ax, data=df[df["label"] == "r"], jitter=True, color='r', alpha=.1)
#sns.stripplot(ax=ax, data=df[df["label"] == "b"], jitter=True, color='b', alpha=.1)

df["combined"] = df["feature"].astype(str) + "_" + df["label"].astype(str)

#sns.stripplot(ax=ax, x="feature", y="diff", data=df, hue="label", jitter=True)
sns.boxplot(ax=ax, x="feature", y="values", data=df, hue="label")
#sns.boxplot(ax=ax, data=df[df["label"] == "r"], color='r')
#sns.boxplot(ax=ax, data=df[df["label"] == "b"], color='b')

plt.xticks(rotation=90)
ax.xaxis.grid(True)
#df_s.plot.box(df["feature_1"], c=df["label"])
plt.tight_layout()
plt.show()