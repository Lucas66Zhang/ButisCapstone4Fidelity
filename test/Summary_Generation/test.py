import sys
sys.path.append('../../src/')
from summary_generation import generate_positive_summary, generate_negative_summary, generate_falsify_summary, generate_paraphrase_summary
import pandas as pd

df = pd.read_csv("../../data/df/31_data_df.csv")
text = df['text_extracted'][2]


positive_result = generate_positive_summary(text)
with open("../../samples/Summary_Generation/positive_result.txt", "w") as f:
    f.write(positive_result)

negative_result = generate_negative_summary(text)
with open("../../samples/Summary_Generation/negative_result.txt", "w") as f:
    f.write(negative_result)

reference_summary = df['Enforcement Summary'][2]
with open("../../samples/Summary_Generation/reference_summary.txt", "w") as f:
    f.write(reference_summary)
    
falsified_summary = generate_falsify_summary(reference_summary)
with open("../../samples/Summary_Generation/falsified_summary.txt", "w") as f:
    f.write(falsified_summary)

paraphrased_summary = generate_paraphrase_summary(reference_summary)
with open("../../samples/Summary_Generation/paraphrased_summary.txt", "w") as f:
    f.write(paraphrased_summary)