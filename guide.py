from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import torch

class Guidance:
    def __init__(self, LM_name, device="cpu"):
        """Initialize a Guidance to restrict the output of a given language model.

        Args:
            LM_name (str): The path in hugging face repo of the language model to use. The LM should support AutoTokenizer.
            device (str, optional): The device to use for the model. Defaults to "cpu".
        """
        self.LM = AutoModelForCausalLM.from_pretrained(LM_name)
        self.tokenizer = AutoTokenizer.from_pretrained(LM_name)
        self.device = torch.device(device)
        if device != "cpu":
            self.LM.to(self.device)

    def generate(self, prompt, options):
        """Generate a text from a prompt, restricted to the options given.

        Args:
            prompt (str): The text to start the generation from.
            options (list[str]): The list of options to restrict the generation to. Currently only supports options of length 1.

        Returns:
            tuple(str, dict[str:float]): The generated text and the dictionary of probabilities of the options.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        last_logits = self.LM(**inputs).logits[:, -1, :]
        options_idx = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
        last_logits_options = [last_logits[:,option_idx].item() for option_idx in options_idx]
        #print('The shape of last_logits_options is: ', len(last_logits_options))
        probs = [math.exp(logit) for logit in last_logits_options]
        #print('The shape of probs is: ', len(probs))
        best_option_idx = probs.index(max(probs))
        best_option = self.tokenizer.decode(options_idx[best_option_idx])
        #prob_dict = dict(zip(options, probs.tolist()))
        sum_probs = sum(probs)
        probs = [number/sum_probs for number in probs]

        return best_option, probs

        
        