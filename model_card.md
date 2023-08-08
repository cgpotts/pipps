---
license: apache-2.0
---

# Model Card for pipp-finder-bert-base-cased

This highly idiosyncratic and specific binary classifier is designed for the sole purpose of helping linguists find instances of the English Preposing in PP (PiPPs) construction in corpora. PiPPs are unbounded dependency constructions like "_Happy though we were with the idea_, we decided not to pursue it". This model does a good job of classifying sentences for whether or not they contain an instance of the construction. 


## Model Details

The model is a fine-tuned `bert-base-cased` model. The fine-tuning data are available as `annotated/pipp-labels.csv` in [this project repository](https://github.com/cgpotts/pipps). All the annotations were done by Christopher Potts for the project "Characterizing English Preposing in PP constructions".

The model outputs `1` if it predicts the input contains a PiPP, else `0`.

### Model Description

- **Developed by:** Christopher Potts
- **Shared by:** Christopher Potts
- **Model type:** Binary classifier
- **Language(s):** English
- **License:** Apache 2.0
- **Finetuned from model:** `bert-base-cased`

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/cgpotts/pipps
- **Paper [optional]:** https://github.com/cgpotts/pipps/blob/main/potts-pipp-paper.pdf


## Uses

The sole purpose of the model is to try to identify sentences containing PiPPs. I assume that one is first filtering sentences using very general regexs, and then this model helps you find the gems as you go through examples by hand.

The model is useless for really anything except this linguistically motivated for task. And, even from the perspective of theoretical linguistics, this is a highly niche application!


## How to Get Started with the Model

See https://github.com/cgpotts/pipps/blob/main/classifiers_usage.ipynb

## Training Details

See https://github.com/cgpotts/pipps/blob/main/classifiers_training.ipynb

## Evaluation

See https://github.com/cgpotts/pipps/blob/main/classifiers_usage.ipynb

## Citation

See https://github.com/cgpotts/pipps
a

## Model Card Authors [optional]

[Christopher Potts](https://web.stanford.edu/~cgpotts/)

## Model Card Contact

[Christopher Potts](https://web.stanford.edu/~cgpotts/) Christopher Potts


