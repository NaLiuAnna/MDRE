#!/bin/bash
mkdir data
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar --exclude='unsup' -xvzf aclImdb_v1.tar.gz -C ./data/
rm -f aclImdb_v1.tar.gz

cd data
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip -j "multinli_1.0.zip" "multinli_1.0/*.jsonl" -d "./multinli_1.0"
rm multinli_1.0.zip

wget https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip
unzip counter-fitted-vectors.txt.zip
rm counter-fitted-vectors.txt.zip