import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
import argparse
import os
import torch
import numpy as np
from pathlib import Path
import glob


####################################################################################
# copying code from fastai
def flip_tensor(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

USE_GPU=True
def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() and USE_GPU else x

class LanguageModelLoader:

    def __init__(self, ds, bs, bptt, backwards=False):
        self.bs,self.bptt,self.backwards = bs,bptt,backwards
        text = sum([o.text for o in ds], [])
        fld = ds.fields['text']
        nums = fld.numericalize([text],device=None if torch.cuda.is_available() else -1)
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        return self

    def __len__(self): return self.n // self.bptt - 1

    def __next__(self):
        if self.i >= self.n-1 or self.iter>=len(self): raise StopIteration
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        res = self.get_batch(self.i, seq_len)
        self.i += seq_len
        self.iter += 1
        return res

    def batchify(self, data):
        nb = data.size(0) // self.bs
        data = data[:nb*self.bs]
        data = data.view(self.bs, -1).t().contiguous()
        if self.backwards: data=flip_tensor(data, 0)
        return to_gpu(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)

class ConcatTextDataset(torchtext.data.Dataset):
    def __init__(self, path, text_field, newline_eos=True, encoding='utf-8', **kwargs):
        fields = [('text', text_field)]
        text = []
        if os.path.isdir(path): paths=glob.glob(f'{path}/*.*')
        else: paths=[path]
        for p in paths:
            for line in open(p, encoding=encoding): text += text_field.preprocess(line)
            if newline_eos: text.append('<eos>')

        examples = [torchtext.data.Example.fromlist([text], fields)]
        super().__init__(examples, fields, **kwargs)


class LanguageData:
    """
    This class provides the entry point for dealing with supported NLP tasks.
    Usage:
    1.  Use one of the factory constructors (from_dataframes, from_text-files) to
        obtain an instance of the class.
    2.  Use the get_model method to return a RNN_Learner instance (a network suited
        for NLP tasks), then proceed with training.
        Example:
            >> TEXT = data.Field(lower=True, tokenize=spacy_tok)
            >> FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
            >> md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=64, bptt=70, min_freq=10)
            >> em_sz = 200  # size of each embedding vector
            >> nh = 500     # number of hidden activations per layer
            >> nl = 3       # number of layers
            >> opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
            >> learner = md.get_model(opt_fn, em_sz, nh, nl,
                           dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
            >> learner.reg_fn = seq2seq_reg
            >> learner.clip=0.3
            >> learner.fit(3e-3, 4, wds=1e-6, cycle_len=1, cycle_mult=2)
    """
    def __init__(self, path, field, trn_ds, val_ds, test_ds, bs, bptt, backwards=False, **kwargs):
        """ Constructor for the class. An important thing that happens here is
            that the field's "build_vocab" method is invoked, which builds the vocabulary
            for this NLP model.
            Also, three instances of the LanguageModelLoader is constructed; one each
            for training data (self.trn_dl), validation data (self.val_dl), and the
            testing data (self.test_dl)
            Args:
                path (str): testing path
                field (Field): torchtext field object
                trn_ds (Dataset): training dataset
                val_ds (Dataset): validation dataset
                test_ds (Dataset): testing dataset
                bs (int): batch size
                bptt (int): back propagation through time
                kwargs: other arguments
        """
        self.bs = bs
        self.path = path
        self.trn_ds = trn_ds; self.val_ds = val_ds; self.test_ds = test_ds
        if not hasattr(field, 'vocab'): field.build_vocab(self.trn_ds, **kwargs)

        self.pad_idx = field.vocab.stoi[field.pad_token]
        self.nt = len(field.vocab)

        self.trn_dl, self.val_dl, self.test_dl = [LanguageModelLoader(ds, bs, bptt, backwards=backwards)
                                                  for ds in (self.trn_ds, self.val_ds, self.test_ds) ]

    @classmethod
    def from_text_files(cls, path, field, train, validation, test=None, bs=64, bptt=70, **kwargs):
        """ Method used to instantiate a LanguageModelData object that can be used for a
            supported nlp task.
        Args:
            path (str): the absolute path in which temporary model data will be saved
            field (Field): torchtext field
            train (str): file location of the training data
            validation (str): file location of the validation data
            test (str): file location of the testing data
            bs (int): batch size to use
            bptt (int): back propagation through time hyper-parameter
            kwargs: other arguments
        Returns:
            a LanguageModelData instance, which most importantly, provides us the datasets for training,
                validation, and testing
        Note:
            The train, validation, and test path can be pointed to any file (or folder) that contains a valid
                text corpus.
        """
        trn_ds, val_ds, test_ds = ConcatTextDataset.splits(
                                    path, text_field=field, train=train, validation=validation, test=test)
        return cls(path, field, trn_ds, val_ds, test_ds, bs, bptt, **kwargs)

############################################################################

# stuff from musicNet project ##############################################

def create_paths():
    PATHS={}
    PATHS['data']=Path('./data/')
    PATHS['critic_data']=Path('./critic_data/')
    PATHS['composer_data']=Path('./composer_data/')
    PATHS['notewise_example_data']=PATHS['data']/'notewise_example_data'
    PATHS['chordwise_example_data']=PATHS['data']/'chordwise_example_data'
    PATHS['chamber_example_data']=PATHS['data']/'chamber_example_data'
    PATHS['models']=Path('./models/')
    PATHS['generator']=PATHS['models']/'generator'
    PATHS['critic']=PATHS['models']/'critic'
    PATHS['composer']=PATHS['models']/'composer'
    PATHS['output']=PATHS['data']/'output'

    for k in PATHS.keys():
        PATHS[k].mkdir(parents=True, exist_ok=True)

    return PATHS

# Unlike language models (which need a tokenizer to recognize don't as similar to 'do not',
# here I have specific encodings for the music, and we can tokenize directly just by splitting by space.
def music_tokenizer(x): return x.split(" ")

def main( test, train, bs, bptt, min_freq):
    """ Loads test/train data, creates a model, trains, and saves it
    Input:
        model_to_load - if continuing training on previously saved model
        model_out - name for saving model
        bs - batch size
        bptt - back prop through time
        em_sz - embedding size
        nh - hidden vector size
        nl - number of LSTM layers
        min_freq - ignore words that don't appear at least min_freq times in the corpus
        dropout_multiplier - 1 defaults to AWD-LSTM paper (the multiplier scales all these values up or down)
        epochs - number of cycles between saves

    Output:
        Trains model, and saves under model_out_light, _med, _full, and _extra
        Models are saved at data/models

    """

    PATHS=create_paths()

    # Check test and train folders have files
    train_files=os.listdir(PATHS["data"]/train)
    test_files=os.listdir(PATHS["data"]/test)
    if len(train_files)<2:
        print(f'Not enough files in {PATHS["data"]/train}. First run make_test_train.py')
        return
    if len(test_files)<2:
        print(f'Not enough files in {PATHS["data"]/test}. First run make_test_train.py, or increase test_train_split')
        return


    TEXT = data.Field(lower=True, tokenize=music_tokenizer)

    FILES = dict(train=train, validation=test, test=test)

    # Build  Language Model Dataset from the training and validation set
    md = LanguageData.from_text_files(PATHS["data"], TEXT, **FILES, bs=bs, bptt=bptt, min_freq=min_freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", dest="bs", help="Batch Size (default 32)", type=int)
    parser.set_defaults(bs=32)
    parser.add_argument("--bptt", dest="bs", help="Back Prop Through Time (default 200)", type=int)
    parser.set_defaults(bptt=200)
    parser.add_argument("--em_sz", dest="em_sz", help="Embedding Size (default 400)", type=int)
    parser.set_defaults(em_sz=400)
    parser.add_argument("--nh", dest="nh", help="Number of Hidden Activations (default 600)", type=int)
    parser.set_defaults(nh=600)
    parser.add_argument("--nl", dest="nl", help="Number of LSTM Layers (default 4)", type=int)
    parser.set_defaults(nl=4)
    parser.add_argument("--min_freq", dest="min_freq", help="Minimum frequencey of word (default 1)", type=int)
    parser.set_defaults(min_freq=1)
    parser.add_argument("--epochs", dest="epochs", help="Epochs per training stage (default 3)", type=int)
    parser.set_defaults(epochs=3)
    parser.add_argument("--prefix", dest="prefix", help="Prefix for saving model (default mod)")
    parser.set_defaults(prefix="mod")
    parser.add_argument("--dropout", dest="dropout", help="Dropout multiplier (default: 1, range 0-5.)", type=float)
    parser.set_defaults(dropout=1)
    parser.add_argument("--load_model", dest="model_to_load", help="Optional partially trained model state dict")
    parser.add_argument("--training", dest="training", help="If loading model, trained level (light, med, full, extra). Default: light")
    parser.set_defaults(training="light")
    parser.add_argument("--test", dest="test", help="Specify folder name in data that holds test data (default 'test')")
    parser.add_argument("--train",dest="train", help="Specify folder name in data that holds train data (default 'train')")
    args = parser.parse_args()

    test = args.test if args.test else "test"
    train = args.train if args.train else "train"

    main(test, train, args.bs, args.bptt, args.min_freq)