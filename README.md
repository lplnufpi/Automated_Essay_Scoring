# Automated Essay Scoring

This research aimed to investigate and propose strategies for AES written in Portuguese through an approach based on the creation and review of an annotated corpus of ENEM essays created from the automated extraction of essays from the internet, as well as the definition of feature set specific to each competency and the development of AES models based on feature engineering methods, Doc2Vec Embeddings and Recurrent Neural Networks  LSTM  to predict an essay grade for each of the five competencies. The results obtained showed that the models based on feature engineering obtained the best results for competencies 1 and 2 with a moderate level of agreement, while for competencies 3, 4 and 5, the model based on LSTM performed better with a moderate level of agreement. The results obtained were validated through an AES web tool developed with the AES models that obtained the best results in this research. With this tool, high school students could have their essays evaluated by the AES models and by a human evaluator. The agreement between the scores of the human evaluator and the AES models reached a moderate level for the competences and a good level for the final score. Although further studies are needed, the results obtained showed that the approach adopted in this research has the potential to be used on a large scale for the automatic assessment of essays written in Portuguese.

## Reference

```
@masterthesis{MarinhoJ,
    author = {Marinho, Jeziel Costa},
    title = {Avaliação automática de redações: Uma abordagem baseada nas competências do ENEM},
    year = {2022},
    school = {Universidade Federal do Piauí}
}

```

## Requirements

- Python (version 3.6 or later)
- `pip install -r requirements.txt`


## Corpus
The corpus used in this research is available [here](https://github.com/lplnufpi/essay-br).
