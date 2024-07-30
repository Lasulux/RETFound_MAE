import pandas as pd

df = pd.read_csv("data/metadata.csv")
normal_df = df[df["N"] == 1]  # both eyes are healthy
normal_df = normal_df[["ID", "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus"]]

normal_left = normal_df.drop("Right-Fundus", axis=1)
normal_right = normal_df.drop("Right-Fundus", axis=1)

normal_left.to_csv("data/normal_left.csv")
normal_right.to_csv("data/normal_right.csv")



