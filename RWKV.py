import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-raven-7b", torch_dtype=torch.float16).to(0)
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-raven-7b")

true_summary = "According to the Commission's complaint, the defendants raised at least $23.4 million from thousands of investors in the Haitian-American community nationwide through a network of purported investment clubs Theodule directs investors to form."
true_text = """The Defendants have engaged in a fraudulent Ponzi scheme primarily targeting the US Haitian community since at least November 2007.This includes net transfers of at least $1.7 million to his personal bank accounts, cash withdrawals of more than $1.5 million and more than $600,000 for apparent personal expenses such as two luxury vehicles, credit card bills, a wedding payment, and a house down payment.21. The investment clubs pool investor funds and send them to Creative Capital for a 90-day period, during which Theodule purportedly trades stocks and options on behalf of the investment club members.Page 2 of 10 
$15.2 million collected from new investors in typical Ponzi scheme fashion.Page 4 of 10 
made millionaires out of a significant number of people in the time it had taken her to decide to invest, and pressured her to liquidate the equity in her home to invest with him.14. Theodule ingratiates himself with investors by claiming he recently decided to offer his investment expertise to help build wealth in the Haitian community.He also tells investors he uses part of his trading profits to fund start-up businesses in the Haitian community, as well as business projects in Haiti and Sierra Leone.15. The Defendants primarily attract investors through word-of-mouth, and Theodule makes his representations during face-to-face meetings in which he touts his ability to double investor funds in just 90 days.18. Since the commencement of the investment scheme, the Defendants have raised more than $23.4 million from thousands of investors nationwide.2, From at least November 2007 to the present, Theodule, directly and through the Companies, has raised at least $23.4 million from thousands of investors in an ongoing fraud and Ponzi scheme targeting mostly Haitian and Haitian-American investors nationwide."""

false_summary = " According to the Commission's complaint, the defendants raised at least $789 billion from thousands of investors in the French community nationwide through a network of purported investment clubs Theodule directs investors to form."

false_text = true_text

true_input_text = f"""
As a compliance officer at a financial institution, you're tasked with evaluating the accuracy of a summary sentence based on its alignment with source sentences from a financial document. Consider the following criteria carefully:

1. The summary accurately reflects the content of the source sentences, especially numerical information.
2. All named entities in the summary are present in the source sentences.
3. Relationships between entities in the summary are consistent with those in the source sentences.
4. The directional flow of relationships among named entities matches between the summary and source sentences.
5. There are no factual discrepancies between the summary and source sentences.
6. The summary does not introduce any entities not found in the source sentences.

Your job is to determine if the summary adheres to these criteria. Answer "Yes" if it does, or "No" if it doesn't.

Summary sentence: ```{true_summary}```

Source sentences: ```{true_text}```

Final Answer (Yes/No only): 
"""
false_input_text = f"""
As a compliance officer at a financial institution, you're tasked with evaluating the accuracy of a summary sentence based on its alignment with source sentences from a financial document. Consider the following criteria carefully:

1. The summary accurately reflects the content of the source sentences, especially numerical information.
2. All named entities in the summary are present in the source sentences.
3. Relationships between entities in the summary are consistent with those in the source sentences.
4. The directional flow of relationships among named entities matches between the summary and source sentences.
5. There are no factual discrepancies between the summary and source sentences.
6. The summary does not introduce any entities not found in the source sentences.

Your job is to determine if the summary adheres to these criteria. Answer "Yes" if it does, or "No" if it doesn't.

Summary sentence: ```{false_summary}```

Source sentences: ```{true_text}```

Final Answer (Yes/No only): 
"""

true_input = tokenizer(true_input_text, return_tensors='pt').to(0)
false_input = tokenizer(false_input_text, return_tensors='pt').to(0)

True_idx = tokenizer.encode("Yes", add_special_tokens=False)[0]
False_idx = tokenizer.encode("No", add_special_tokens=False)[0]

with torch.no_grad():
    true_output = model(**true_input)
true_logits = true_output.logits
prob = torch.softmax(true_logits[:, -1, :], dim=1)
yes_prob = prob[:, True_idx].item()
no_prob = prob[:, False_idx].item()
print(yes_prob, no_prob)
print(yes_prob > no_prob)

with torch.no_grad():
    false_output = model(**false_input)
false_logits = false_output.logits
false_prob = torch.softmax(false_logits[:, -1, :], dim=1)
yes_false_prob = false_prob[:, True_idx].item()
no_false_prob = false_prob[:, False_idx].item()
print(yes_false_prob, no_false_prob)
print(yes_false_prob > no_false_prob)