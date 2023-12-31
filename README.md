<h1 align='center'>
  Benchmarking LLM 
</h1>

<div align="center">
    <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Lucas66Zhang/ButisCapstone4Fidelity?style=for-the-badge">
    <a href="https://github.com/Lucas66Zhang/ButisCapstone4Fidelity/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Lucas66Zhang/ButisCapstone4Fidelity?style=for-the-badge"></a>
    <a href="https://github.com/Lucas66Zhang/ButisCapstone4Fidelity/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/Lucas66Zhang/ButisCapstone4Fidelity?style=for-the-badge"></a>
  <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/Lucas66Zhang/ButisCapstone4Fidelity?style=for-the-badge">
    <a href="https://github.com/Lucas66Zhang/ButisCapstone4Fidelity/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/Lucas66Zhang/ButisCapstone4Fidelity?style=for-the-badge">

</div>
<br>

## About The Project
Our project introduces a metric designed to evaluate the quality of textual summaries. This metric is pivotal in fields like finance, where precise information synthesis is critical.

* **Quality Discrimination**: Distinguishes effectively between superior and inferior summaries, ensuring clear differentiation in their factual accuracies.
* **Factual Accuracy Measurement**: Detects and quantifies any factual deviations, assigning lower scores to less accurate summaries.
* **Detail-Oriented Assessment**: Provides comprehensive evaluations, focusing on how well the summary captures the essence and details of the original text.
  
This metric is not merely a tool for evaluation; it's a step towards enhancing the integrity of information processing in sectors where factual accuracy is non-negotiable.

## Framework
**Named Entity Comparison**: Extract and compare financial-related named entities in texts. Analyzes and visualizes named entity accuracy and presence in summaries versus original texts.
<p align="center">
    <img alin = "center" src="./res/NER_Framework.jpg" style="width:45%">
</p>


**Sentence-Level-based Summary Checking**: Applies LLMs to check the consistency between the summary and the original text sentence by sentence. Highlights and identifies inconsistencies between the summary and the original text for in-depth analysis.
<p align="center">
    <img alin = "center" src="./res/LLM_Assisted_Framework.jpg" style="width:45%">
</p>

## Direcroty Tree
```
│   .gitignore
│   LICENSE.txt
│   README.md
│
├───config
│       config.sh
│       requirements.txt
│
├───data
│       10summary_with_result.csv
│       falsified_summary.csv
│       falsified_summary_level.csv
│       final_version_cropped_first1000.csv
│       final_version_withouttext.csv
│
├───doc
│   ├───About_Us
│   │       Team's Bio.pdf
│   │
│   ├───Academic Paper
│   │       5054_factuality_enhanced_language_m.pdf
│   │       Evaluating Factuality.pdf
│   │       Evaluating the Factual Consistency.pdf
│   │
│   ├───Project Description
│   │       Benchmarking LLM .pdf
│   │       CAPSTONE PROJECT PROPOSAL Fidelity Summarization Metrics.pdf
│   │
│   └───Report
│           Capstone Project Initial Due Diligence Report.pdf
│           F23_Fidelity_Benchmarking LLM_1st_report.pdf
│           F23_Fidelity_Benchmarking LLM_final_report.pdf
│           F23_Fidelity_BenchmarkLLM_poster.pdf
│           Project Proposal.pdf
│
├───res
│   │   10levels.svg
│   │   good_to_bad.svg
│   │   LLM_Assisted_Framework.jpg
│   │   NER_Framework.jpg
│   │
│   └───Baseline
│           Boxplot_for_Scores.png
│
├───samples
│       documents_extraction.ipynb
│       presentation.ipynb
│       summary_level_with_result.csv
│
├───src
│   │   Bart.py
│   │   PaLM.py
│   │   pipeline.py
│   │   summary_generation.py
│ 
│
└───test
    ├───Data_Pipeline
    └───Summary_Generation
            bart.ipynb
            llama2.ipynb
            PaLM2.ipynb
            test.py
```

## Getting Started
<div align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
</div>
<be>

### Dependencies
```
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
![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)
#### 1. Environment setup
##### Setup with python virtual environment
```bash ./config/config.sh```

##### Setup with conda
```bash conda install --file ./config/requirements.txt```

#### 2. OpenAI API setup
```python
import sys
sys.path.append('../src/')
import pipeline
os.environ['OPENAI_API_KEY'] = 'Your OpenAi API Key'
```



## Usage
<img alt="Jupyter Notebook" src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter">

The data extraction process is in [documents_extraction](./samples/documents_extraction.ipynb)

You can also find the demo and result compare with baseline metrics in [presentation](./samples/presentation.ipynb).


## Report
![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)
* **Initial Due Deiligence Report**: [Initial Due Deiligence Report](./doc/Report/Capstone%20Project%20Initial%20Due%20Diligence%20Report.pdf)
* **Project Proposal**: [Project Proposal](./doc/Report/Project%20Proposal.pdf)
* **1st Milestone Report**: [1st Milestone Report](./doc/Report/F23_Fidelity_Benchmarking%20LLM_1st_report.pdf)
*  **Final Report**: [Final Report](./doc/Report/F23_Fidelity_Benchmarking%20LLM_final_report.pdf)
*  **Poster**: [Poster](./doc/Report/F23_Fidelity_BenchmarkLLM_poster.pdf)


## License

[![Generic badge](https://img.shields.io/badge/License-Apache%202.0-Green?style=for-the-badge)](./LICENSE.txt)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLucas66Zhang%2FButisCapstone4Fidelity&count_bg=%2379C83D&title_bg=%23555555&icon=proto-dot-io.svg&icon_color=%23E7E7E7&title=Repo+Viewer+Count&edge_flat=true)](https://hits.seeyoufarm.com)

## [About us](./doc/About_US/Team's%20Bio.pdf)
### Group Members
#### Cong Chen (cc4887)
<div align="Left">
    <a href="mailto: cc4887@columbia.edu"><img alt="Email" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
    <a href="https://github.com/Cong991"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

#### Longxiang Zhang (lz2869)
<div align="Left">
    <a href="mailto: lz2869@columbia.edu"><img alt="Email" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
    <a href="https://github.com/Lucas66Zhang"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

#### Ruolan Lin (rl3312)
<div align="Left">
    <a href="mailto: rl3312@columbia.edu"><img alt="Email" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
    <a href="https://github.com/Ruolan0806"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

#### Taichen Zhou (tz2555)
<div align="Left">
    <a href="mailto: tz2555@columbia.edu"><img alt="Email" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
    <a href="https://github.com/tzhou19"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

#### Yichen Huang (yh3550) - Team Captain
<div align="Left">
    <a href="mailto: yichen.huang@columbia.edu"><img alt="Email" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
    <a href="https://github.com/yichuang25"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a>
    <a href="https://www.linkedin.com/in/huangyichen/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>
</div>
<be>

### Fidelity Memtors
Lilli Ann Rowan, Indraneel Biswas, Michael Threlfall, and Diana Kulmizev

### Instructor/CA
Adam Kelleher



