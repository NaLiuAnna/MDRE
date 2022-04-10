### About The Project
Source code for the paper: Detecting Textual Adversarial Examples Based on Distributional Characteristics of Data Representations.

Requirements:
* numpy(1.19.5)
* transformers(4.1.1)
* pandas(1.1.5)
* spacy(3.0.5)
* torch(1.7.1)
* nltk(3.5)
* scipy(1.4.1)
* scikit-learn(0.24.0)
* tqdm(4.62.3)
* torchtext(0.1.1)
* editdistance(0.5.3)

The code is built in Python 3.6.3. 
To install all required packages, please run
  
   `pip install -r requirements.txt`
   
and run followings on GPU:

`git clone https://github.com/marcotcr/OpenNMT-py`

`cd OpenNMT-py/`

`python setup.py install`

`cd ..`

### Obtaining the data
To download the IMDB, MultiNLI datasets and counter-fitted vectors

    bash ./download.sh

### Fine-tune transformers models/representation learning models

#### IMDB dataset
`python fine_tune.py  --model-name bert --dataset-name IMDB --dataset-path ./data/aclImdb --max-length 512`

change the --model-name to roberta, xlnet, bart to fine-tune RoBERTa, XLNet, BART models.

#### MultiNLI dataset
`python fine_tune.py  --model-name bert --dataset-name Mnli --dataset-path ./data/multinli_1.0 --max-length 256`

change the --model-name to roberta, xlnet, bart to fine-tune RoBERTa, XLNet, BART models.

### Generate textual adversarial examples
This project uses character-level, word-level, and phrase-level textual adversarial examples, with allowable values for
--attack-class argument of 'typo', 'synonym', or 'seas', with --topk 30 for the MultiNLI dataset SEAs attack.

To generate character-level/typo textual adversarial examples for the IMDB dataset:

`python generate_adv.py --dataset-name IMDB --dataset-path ./data/aclImdb --attack-class typo --max-length 512 --batch 0 --boxsize 25`

--boxsize is the total number of batches for test examples, --batch is the number of batch.
This command will generate adversarial examples for the top 1,000 IMDB test examples.

To generate character-level/typo textual adversarial examples for the MultiNLI dataset:

`python generate_adv.py --dataset-name Mnli --dataset-path ./data/multinli_1.0 --attack-class typo --max-length 256 --boxsize 10 --batch 0`

* before generate word-level/synonym adversarial examples, please run:

  `python get_neighbours.py --dataset-name IMDB --dataset-path ./data/aclImdb --max-length 512`
  
  or

    `python get_neighbours.py --dataset-name Mnli --dataset-path ./data/multinli_1.0 --max-length 256`

* before generate phrase-level/seas adversarial examples, please download and unpack the [translation models](https://drive.google.com/open?id=1b2upZvq5kM0lN0T7YaAY30xRdbamuk9y) into the translation_models folder.

### Detect textual adversarial examples
We use MDRE, adapted LID, FGWS, and a language model as detection classifiers, with --detect argument choices of 'mdre', 'lid', 'fgws', and 'language_model'.

To detect character-level adversarial examples using MDRE on the IMDB dataset, please run:

`python detect.py --dataset-name IMDB --dataset-path ./data/aclImdb --attack-class typo --max-length 512 --batch-size 32 --detect mdre`

To detect character-level adversarial examples using MDRE on the MultiNLI dataset, please run:

`python detect.py --dataset-name Mnli --dataset-path ./data/multinli_1.0 --attack-class typo --max-length 256 --batch-size 32 --detect mdre`
