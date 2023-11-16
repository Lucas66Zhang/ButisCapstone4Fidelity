



true_summary = "According to the Commission's complaint, the defendants raised at least $23.4 million from thousands of investors in the Haitian-American community nationwide through a network of purported investment clubs Theodule directs investors to form."
true_text = """The Defendants have engaged in a fraudulent Ponzi scheme primarily targeting the US Haitian community since at least November 2007.This includes net transfers of at least $1.7 million to his personal bank accounts, cash withdrawals of more than $1.5 million and more than $600,000 for apparent personal expenses such as two luxury vehicles, credit card bills, a wedding payment, and a house down payment.21. The investment clubs pool investor funds and send them to Creative Capital for a 90-day period, during which Theodule purportedly trades stocks and options on behalf of the investment club members.Page 2 of 10 
$15.2 million collected from new investors in typical Ponzi scheme fashion.Page 4 of 10 
made millionaires out of a significant number of people in the time it had taken her to decide to invest, and pressured her to liquidate the equity in her home to invest with him.14. Theodule ingratiates himself with investors by claiming he recently decided to offer his investment expertise to help build wealth in the Haitian community.He also tells investors he uses part of his trading profits to fund start-up businesses in the Haitian community, as well as business projects in Haiti and Sierra Leone.15. The Defendants primarily attract investors through word-of-mouth, and Theodule makes his representations during face-to-face meetings in which he touts his ability to double investor funds in just 90 days.18. Since the commencement of the investment scheme, the Defendants have raised more than $23.4 million from thousands of investors nationwide.2, From at least November 2007 to the present, Theodule, directly and through the Companies, has raised at least $23.4 million from thousands of investors in an ongoing fraud and Ponzi scheme targeting mostly Haitian and Haitian-American investors nationwide."""

false_summary = " According to the Commission's complaint, the defendants raised at least $789 million from thousands of investors in the Haitian-American community nationwide through a network of purported investment clubs Theodule directs investors to form."
false_text = """The Defendants have engaged in a fraudulent Ponzi scheme primarily targeting the US Haitian community since at least November 2007.This includes net transfers of at least $1.7 million to his personal bank accounts, cash withdrawals of more than $1.5 million and more than $600,000 for apparent personal expenses such as two luxury vehicles, credit card bills, a wedding payment, and a house down payment.21. The investment clubs pool investor funds and send them to Creative Capital for a 90-day period, during which Theodule purportedly trades stocks and options on behalf of the investment club members.Page 2 of 10 
$15.2 million collected from new investors in typical Ponzi scheme fashion.Page 4 of 10 
made millionaires out of a significant number of people in the time it had taken her to decide to invest, and pressured her to liquidate the equity in her home to invest with him.14. Theodule ingratiates himself with investors by claiming he recently decided to offer his investment expertise to help build wealth in the Haitian community.He also tells investors he uses part of his trading profits to fund start-up businesses in the Haitian community, as well as business projects in Haiti and Sierra Leone.15. The Defendants primarily attract investors through word-of-mouth, and Theodule makes his representations during face-to-face meetings in which he touts his ability to double investor funds in just 90 days.18. Since the commencement of the investment scheme, the Defendants have raised more than $23.4 million from thousands of investors nationwide.2, From at least November 2007 to the present, Theodule, directly and through the Companies, has raised at least $23.4 million from thousands of investors in an ongoing fraud and Ponzi scheme targeting mostly Haitian and Haitian-American investors nationwide."""

true_input_text = f"""Evaluate the compliance of a summary sentence derived from a set of sentences in a financial document. Adhere to the following verification standards:
1. Entity consistency: Check that all named entities in the summary are extracted from the source.
2. Relationship verification: Confirm that relationships between entities in the summary are present and correctly depicted in the source.
3. Directionality check: Ensure that the direction of relationships between entities in the summary matches those in the source.
4. Factual integrity: Ascertain that the summary is free from factual errors when compared to the source.
5. Entity authenticity: Confirm that the summary does not create non-existent entities.

Based on these criteria, determine if the summary sentence is a faithful representation of the source sentences. Respond with "True" if the summary complies with all standards, or "False" if it does not.

Summary sentence: {true_summary}

Source sentences: {true_text}"""

false_input_text = f"""Evaluate the compliance of a summary sentence derived from a set of sentences in a financial document. Adhere to the following verification standards:
1. Entity consistency: Check that all named entities in the summary are extracted from the source.
2. Relationship verification: Confirm that relationships between entities in the summary are present and correctly depicted in the source.
3. Directionality check: Ensure that the direction of relationships between entities in the summary matches those in the source.
4. Factual integrity: Ascertain that the summary is free from factual errors when compared to the source.
5. Entity authenticity: Confirm that the summary does not create non-existent entities.

Based on these criteria, determine if the summary sentence is a faithful representation of the source sentences. Respond with "True" if the summary complies with all standards, or "False" if it does not.

Summary sentence: {false_summary}

Source sentences: {false_text}"""



import openai
openai.api_key = "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"


completions = openai.Completion.create(
    engine="davinci",
    prompt=true_input_text,
    max_tokens=1,
    n=1,
    logprobs=10,
)

text = completions.choices[0].text.strip()
logprobs = completions.choices[0].logprobs.token_logprobs
tokens = logprobs.keys()
probs = logprobs.values()

for token, prob in zip(tokens, probs):
    print(f"Token: {token}, Log Probability: {prob}")