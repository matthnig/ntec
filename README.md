# ntec

[![PyPI version](https://badge.fury.io/py/ntec.svg)](https://badge.fury.io/py/ntec)

## 1. What is ntec?

ntec is short for '**N**ame **T**o **E**thnicity **C**lassification' and is a Python-based framework for ethnicity classification based on peoples' names. It was first introduced and used in my paper [Moving On - Investigating Inventors' Ethnic Origins Using Supervised learning](xxxx), where you can find methodological details of the main classifier and its training data. In short, ntec builds on a trained artificial neural network that uses a name's letters to predict its ethnic origin.

## 2. Installation

### Step 1: Install tensorflow

ntec runs on top of keras and tensorflow (version 2.5 at the time of development). Hence, these packages have to be installed first. The [developer homepage](https://www.tensorflow.org/install) provides the details for the installation process (using a virtual environment is recommended). Afterwards, verify the installation of tensorflow as suggested by the devleopers:

```python
import tensorflow as tf
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
```

### Step 2: Install ntec
ntec can be installed via pip:

```bash
pip install ntec
```

## 3. Quick start

ntec currently builds on the classifier presented in the article [Moving On - Investigating Inventors' Ethnic Origins Using Supervised learning](xxxx), which is labelled accordingly as the 'joeg'-classifier. You can load this classifier, check its parameters and recognized ethnic origin classes using the code below.

```python
import ntec

# load the classifier
classifier = ntec.Classifier("joeg") # initialize the 'joeg' classifier

# check the parameters of the classifier
classifier.params["seq_max"]
# 30
classifier.params["n_chars"]
# 27
classifier.params["char_dict"]
# ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

# check the ethnic origin classes recognized by the classifier
classifier.classes
# {'0': 'AngloSaxon', '1': 'Arabic', '2': 'Balkans', '3': 'Chinese', '4': 'East-European', '5': 'French', '6': 'German', '7': 'Hispanic-Iberian', '8': 'Indian', '9': 'Italian', '10': 'Japanese', '11': 'Korean', '12': 'Persian', '13': 'Scandinavian', '14': 'Slavic-Russian', '15': 'South-East-Asian', '16': 'Turkish', '17': 'Dutch'}
```

As specified by the `classifier.params["char_dict"]` attribute, the 'joeg' classifier only accepts names that entirely consist of either ASCII lowercased letters or whitespace. Hence, any name whose ethnic origins are to be precited by the chosen classifier has to be cleaned of non-ASCII letters first. ntec offers the function `clean_name()` for this task, which is demonstarted by the sample code below.

```python
# single name cleaning:
cleaned_name = ntec.clean_name(name = "ruud van niste'\lrooy")
print(cleaned_name)
# 'ruud van nistelrooy'

# multiple name cleaning:
# define example names (including some non-ascii characters, uppercase letters and digits):
names = [
    "tony aDams", "Mustapha Hadji", "sinisa mihajlovic", 
    "Sun Jihai", "ruud van niste'\lrooy", "tomasz rosicky ", 
    "didier d_eschamps", "oliver kahn", "gabriel batistuta", 
    "Sunil Chhetri", "paolo maldini3", "Shunsuke naka\\mura", 
    "Ji-Sung Park", "ali daei xasfdhkljhasdfhakghfiugasfd", "henrik larsson", 
    "Andrey Arshavin", "Teerasil Dangda", "hakan sükür",  
     ]
cleaned_names = [ntec.clean_name(name) for name in names]
print(cleaned_names)
# ['tony adams', 'mustapha hadji', 'sinisa mihajlovic', 'sun jihai', 'ruud van nistelrooy', 
# 'tomasz rosicky', 'didier deschamps', 'oliver kahn', 'gabriel batistuta', 'sunil chhetri',
# 'paolo maldini', 'shunsuke nakamura', 'jisung park', 'ali daei xasfdhkljhasdfhakghfiugasfd',
# 'henrik larsson', 'andrey arshavin', 'teerasil dangda', 'hakan sukur']
```

After cleaning, names must be encoded to a form that can be processed by the classifier. This encoding step is performed using the classifier's `classifier.encode_name()`-method as shown below. First, the `classifier.encode_name()` method automatically ensures that the lenghth of an input name does not surpass the classifier's `classifier.params["seq_max"]` attribute (i.e., longer names such as "ali daei xasfdhkljhasdfhakghfiugasfd" in the example will be cut to this length). Second, it then transforms a the cleaned name according to the `classifier.params` attribute to a 2D-numpy.array of shape `(seq_max`, `n_chars + 1)`.

```python
import numpy as np

# single name encoding:
encoded_name = classifier.encode_name(name = cleaned_name)
encoded_name.shape
# (30, 28)

# multiple name encoding
encoded_names = np.array([classifier.encode_name(name) for name in cleaned_names])
encoded_names.shape
# (18, 30, 28)
```

Subsequetly, the encoded names can be sent to the classifier, which then predicts corresponding ethnic origins.

```python
import pandas as pd

# single name origin prediction:
origin_pred = classifier.predict_origins(x = encoded_name, output = "classes")
pd.concat([pd.Series(cleaned_name, name = "cleaned_name"), origin_pred], axis = 1)
#           cleaned_name ethnic_origin
# 0  ruud van nistelrooy         Dutch

# multiple name origin prediction:
origin_pred = classifier.predict_origins(np.array(encoded_names), output = "classes")
pd.concat([pd.Series(cleaned_names, name = "cleaned_name"), origin_pred], axis = 1)
#                             cleaned_name     ethnic_origin
# 0                             tony adams        AngloSaxon
# 1                         mustapha hadji            Arabic
# 2                      sinisa mihajlovic           Balkans
# 3                              sun jihai           Chinese
# 4                    ruud van nistelrooy             Dutch
# 5                         tomasz rosicky     East-European
# 6                       didier deschamps            French
# 7                            oliver kahn            German
# 8                      gabriel batistuta  Hispanic-Iberian
# 9                          sunil chhetri            Indian
# 10                         paolo maldini           Italian
# 11                     shunsuke nakamura          Japanese
# 12                           jisung park            Korean
# 13  ali daei xasfdhkljhasdfhakghfiugasfd           Persian
# 14                        henrik larsson      Scandinavian
# 15                       andrey arshavin    Slavic-Russian
# 16                       teerasil dangda  South-East-Asian
# 17                           hakan sukur           Turkish
```

## Contact
If you have any feedback or questions, please contact me at matthias.niggli@gmx.ch

## Citation
Please cite appropriately as:

Matthias Niggli (2023), ‘Moving On’—investigating inventors’ ethnic origins using supervised learning, *Journal of Economic Geography*, lbad001, [https://doi.org/10.1093/jeg/lbad001](https://doi.org/10.1093/jeg/lbad001).

BibTex:
```
@article{niggli2023,
    author = {Niggli, Matthias},
    title = "{‘Moving On’—investigating inventors’ ethnic origins using supervised learning}",
    journal = {Journal of Economic Geography},
    year = {2023},
    month = {01},
    abstract = "{Patent data provides rich information about technical inventions, but does not disclose the ethnic origin of inventors. In this article, I use supervised learning techniques to infer this information. To do so, I construct a dataset of 96′777 labeled names and train an artificial recurrent neural network with long short-term memory (LSTM) to predict ethnic origins based on names. The trained network achieves an overall performance of 91.4\\% across 18 ethnic origins. I use this model to predict and investigate the ethnic origins of 2.68 million inventors and provide novel descriptive evidence regarding their ethnic origin composition over time and across countries and technological fields. The global ethnic origin composition has become more diverse over the last decades, which was mostly due to a relative increase of Asian origin inventors. Furthermore, the prevalence of foreign-origin inventors is especially high in the USA, but has also increased in other high-income economies. This increase was mainly driven by an inflow of non-Western inventors into emerging high-technology fields for the USA, but not for other high-income countries.}",
    issn = {1468-2702},
    doi = {10.1093/jeg/lbad001},
    url = {https://doi.org/10.1093/jeg/lbad001},
    note = {lbad001},
    eprint = {https://academic.oup.com/joeg/advance-article-pdf/doi/10.1093/jeg/lbad001/48958974/lbad001.pdf},
}
```
