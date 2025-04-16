import pandas as pd
import ast

df = pd.read_csv("output_rdfLysosomes_plate1.csv")

df["LysosomeRDF"] = df["LysosomeRDF"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

for _, row in df.iterrows():
    print(row["LysosomeRDF"][0:5])

