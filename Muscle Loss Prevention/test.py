import pandas as pd

df = pd.read_csv("cleaned_fitness_dataset.csv")

X = df.drop("muscle_loss_risk", axis=1)

print(X.columns)