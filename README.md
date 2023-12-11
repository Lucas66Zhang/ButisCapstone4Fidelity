# Benchmarking LLM

## About The Project
Our project introduces a metric designed to evaluate the quality of textual summaries. This metric is pivotal in fields like finance, where precise information synthesis is critical.

* **Quality Discrimination**: Distinguishes effectively between superior and inferior summaries, ensuring clear differentiation in their factual accuracies.
* **Factual Accuracy Measurement**: Detects and quantifies any factual deviations, assigning lower scores to less accurate summaries.
* **Detail-Oriented Assessment**: Provides comprehensive evaluations, focusing on how well the summary captures the essence and details of the original text.
  
This metric is not merely a tool for evaluation; it's a step towards enhancing the integrity of information processing in sectors where factual accuracy is non-negotiable.

## Framework
**Named Entity Comparison**: Extract and compare financial-related named entities in texts. Analyzes and visualizes named entity accuracy and presence in summaries versus original texts.
<p align="center">
    <img alin = "center" src="./res/NER_Framework.jpg" style="width:50%">
</p>


**Sentence-Level-based Summary Checking**: Applies LLMs to check the consistency between the summary and the original text sentence by sentence. Highlights and identifies inconsistencies between the summary and the original text for in-depth analysis.
<p align="center">
    <img alin = "center" src="./res/LLM_Assisted_Framework.jpg" style="width:50%">
</p>


## Getting Started
### Dependencies
```json
python==3.10.0
ipython==8.15.0
nltk==3.8.1
numpy==1.24.3
openai==1.3.7
pandas==1.5.3
python-dotenv==1.0.0
rouge_score==0.1.2
scikit_learn==1.2.2
sentence_transformers==2.2.2
spacy==3.7.2
stanza==1.6.1
```

### Configuration
#### Setup with python virtual environment
```bash ./config/config.sh```

#### Setup with conda
```bash conda install --file ./config/requirements.txt```


## Usage
The data extraction process is in [documents_extraction](./samples/documents_extraction.ipynb)

You can also find the demo and result compare with baseline metrics in [presentation](./samples/presentation.ipynb).


## Report
* **Initial Due Deiligence Report**: [Initial Due Deiligence Report](./doc/Report/Capstone%20Project%20Initial%20Due%20Diligence%20Report.pdf)
* **Project Proposal**: [Project Proposal](./doc/Report/Project%20Proposal.pdf)
* **1st Milestone Report**: [1st Milestone Report](./doc/Report/F23_Fidelity_Benchmarking%20LLM_1st_report.pdf)
*  **Final Report**: [Final Report](./doc/Report/F23_Fidelity_Benchmarking%20LLM_final_report.pdf)
*  **Poster**: [Poster](./doc/Report/F23_Fidelity_BenchmarkLLM_poster.pdf)


## License
Distributed under the Apache-2.0 License. See `LICENSE.txt` for more information.

## [About us](./doc/About_US/Team's%20Bio.pdf)
Yichen Huang - yichen.huang@columbia.edu

Taichen Zhou - tz2555@columbia.edu

Cong Chen - cc4887@columbia.edu

Ruolan Lin - rl3312@columbia.edu

Longxiang Zhang - lz2869@columbia.edu