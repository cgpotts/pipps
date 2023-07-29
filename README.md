# Characterizing English Preposing in PP constructions

Christopher Potts

## Paper

Potts, Christopher. 2023. [Characterizing English Preposing in PP constructions](potts-pipp-paper.pdf). To be submitted to _English language and linguistic theory: A tribute to Geoff Pullum_.

## Contents

This code is intended to be run with Python 3.9.

* `books.ipynb`: High-level stats for BookCorpusOpen, and sampling examples to annotate.
* `c4.ipynb`: High-level stats for C4, and sampling examples to annotate.
* `annotated/`: Hand-annotated samples.
* `frequency_estimates.ipynb`: PiPP frequency estimates for BookCorpusOpen and C4 based on the annotated samples.
* `materials.txt`: Sentences used in the LLM experiments.
* `materials_check.ipynb`: Notebook to facilitate checking examples in `materials.txt` in different variants.
* `experiments_wh_effects.ipynb`: Notebook for running the filler-gap experiments and summarizing the results.
* `experiments_prep_effects.ipynb`: Notebook for running the prepostional head experiments.
* `experiments_llm_transformations.ipynb`: Notebook for the exploratory pilot experiments seeing whether models can map PPs to PiPPs via few-shot in-context learning.
* `materials-stress-test.csv`: Additional hard examples for the transformations experiment.
* `results/`: All experimental results and visualizations.
* `pipp.mplstyle`: Matplotlib style file for the project.
