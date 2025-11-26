import pandas as pd

def build_csv(path1="extremist_1500.csv", path2="extremist.csv", path3="normal_1500.csv", path4="non_extremist.csv"):

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)

    df1['extreme'] = 1
    df2['extreme'] = 1
    df3['extreme'] = 0
    df4['extreme'] = 0

    df1['pre_2018'] = 1
    df2['pre_2018'] = 0
    df3['pre_2018'] = 1
    df4['pre_2018'] = 0

    df = pd.concat([df1, df2, df3, df4])

    df.to_csv("tweets.csv", index=False)


if __name__ == "__main__":
    build_csv()