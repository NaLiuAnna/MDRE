# Generating character-level(typo), word-level(synonym replacement), or phrase-level(SEAs) textual adversarial
# examples, and using the BERT base model as the target model
import os
import random
import argparse
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from models import Classifier
from attack import Typo, SynonymsReplacement, SEAs
from utils import IMDBDataset, MnliDataset
from utils import bert_params

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, default=None, required=True, choices=['IMDB', 'Mnli'],
                    help='Which test set to perturb to generate adversarial examples.')
parser.add_argument('--dataset-path', type=str, default=None, required=True,
                    choices=['./data/aclImdb', './data/multinli_1.0'], help='The directory of the dataset.')
parser.add_argument('--attack-class', type=str, default=None, required=True, choices=['typo', 'synonym', 'seas'],
                    help='Attack method to generate adversarial examples.')
parser.add_argument('--max-length', type=int, default=None, required=True, choices=[512, 256, 128],
                    help='The maximum sequence length.')
parser.add_argument('--topk', type=int, default=100, required=False, choices=[100, 30],
                    help='A parameter used in SEAs attack decides how many potential adversarial examples to choose.')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size used for BERT prediction.')
parser.add_argument('--random-seed', type=int, default=38, help='The random seed value.')
parser.add_argument('--batch', type=int, help='batch number in all batches of test data for distributed data parallel.')
parser.add_argument('--boxsize', type=int, help='How many batches are used for distributed data parallel?')
args = parser.parse_args()

# set a random seed value all over the place to make this reproducible
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

# check if there's a GPU
if torch.cuda.is_available():
    # set the device to the GPU
    device = torch.device('cuda')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

model_params = bert_params
batch_size = args.batch_size
max_len = args.max_length

# load dataset
data_processors = {
    'Mnli': MnliDataset,
    'IMDB': IMDBDataset
}
dataset = data_processors[args.dataset_name](args.dataset_path, 0.2)

# load BERT model and tokenizer
output_dir = os.path.join('./output', args.dataset_name, 'bert')
model = Classifier(dataset.num_labels, **model_params)
model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt'), map_location=device))
model.to(device)
tokenizer = BertTokenizer.from_pretrained(output_dir)


def get_predictions(texts):
    """
    Obtaining the model prediction of `texts`
    :param texts: input of the BERT model
    :return: predictions and outputs of softmax which represent the probability distribution over classes
    """
    model.eval()
    n_batch = ceil(len(texts) / batch_size)
    preds = []
    softmax_list = []
    for batch in range(n_batch):
        begin_idx = batch_size * batch
        end_idx = min(batch_size * (batch + 1), len(texts))
        b_texts = texts[begin_idx: end_idx]
        text = np.asarray(b_texts)[:, 0].tolist()
        text_pair = np.asarray(b_texts)[:, 1].tolist()

        inputs = tokenizer(text=text,
                           text_pair=text_pair if text_pair[0] else None,
                           return_tensors='pt',
                           max_length=max_len,
                           truncation=True,
                           padding='max_length',
                           return_attention_mask=True).to(device)

        with torch.no_grad():
            logits, _, _ = model(inputs['input_ids'], inputs['attention_mask'])
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        softmax_list.append(F.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(softmax_list, axis=0)


# initialize attack instance
attack_processors = {
    'typo': Typo,
    'synonym': SynonymsReplacement,
    'seas': SEAs
}
attack_params = {
    'typo': [max_len, dataset.num_labels],
    'synonym': [args.dataset_name, max_len, dataset.num_labels],
    'seas': [dataset.num_labels, args.topk]
}
attack = attack_processors[args.attack_class](get_predictions, *attack_params[args.attack_class])

# generate adversarial examples
num_texts = len(dataset.test_y)
boxsize = args.boxsize
begin_idx = -(num_texts // -boxsize) * args.batch
end_idx = min(-(num_texts // -boxsize) * (args.batch + 1), num_texts)
result = attack.generate(dataset.test_text[begin_idx: end_idx], dataset.test_text_pair[begin_idx: end_idx],
                         dataset.test_y[begin_idx: end_idx])

# create directory if not exist
if not os.path.exists(os.path.join(output_dir, args.attack_class)):
    os.makedirs(os.path.join(output_dir, args.attack_class))
# save adversarial examples
result.to_csv(os.path.join(output_dir, args.attack_class, f'{args.attack_class}_adv_{args.batch}.csv'), index=False)

print('Done!')