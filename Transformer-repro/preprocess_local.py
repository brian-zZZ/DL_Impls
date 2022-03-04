''' Handling the data io '''
import os
import argparse
import logging
import dill as pickle
from tqdm import tqdm
from itertools import chain
import sys
import codecs
import tarfile
import torchtext.legacy.data
import torchtext.legacy.datasets
from torchtext.legacy.datasets import TranslationDataset
import transformer.Constants as Constants
from learn_bpe import learn_bpe
from apply_bpe import BPE


__author__ = "Brian Zhang, College of AI, UCAS"


# Notice: Must specify the 'dir' locally.
_TRAIN_DATA_SOURCE = {
    "dir": "./data/training-parallel-nc-v12.tgz",
    "trg": "news-commentary-v12.de-en.en",
    "src": "news-commentary-v12.de-en.de"}
_VAL_DATA_SOURCE = {
    "dir": "./data/dev.tgz",
    "trg": "newstest2013.en",
    "src": "newstest2013.de"}
_TEST_DATA_SOURCE ={
    "dir": "./data/dev.tgz",
    "trg": "newstest2012.en",
    "src": "newstest2012.de"}


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def file_exist(dir_name, file_name):
    '''Return file_path if src or trg file exist, else return None'''
    # os.walk()逐层游走，迭代生成：当前路径, 当前路径下的子文件夹list, 当前路径下的文件list
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def extract_raw_files(raw_dir, source_dict):
    '''Extract raw files from zip-files, return a dict containing src/trg path'''
    raw_files = { "src": [], "trg": [], }

    compressed_file = source_dict['dir']
    src_filename, trg_filename = source_dict['src'], source_dict['trg']

    # Check whether raw files have been extract
    src_path = file_exist(raw_dir, src_filename)
    trg_path = file_exist(raw_dir, trg_filename)
    if src_path and trg_path:
        sys.stderr.write(f"Already extracted {compressed_file.split('/')[-1]}.\n")
        raw_files["src"].append(src_path)
        raw_files["trg"].append(trg_path)
        return raw_files

    sys.stderr.write(f"Extracting {compressed_file.split('/')[-1]}.\n")
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        corpus_tar.extractall(raw_dir)

    src_path = file_exist(raw_dir, src_filename)
    trg_path = file_exist(raw_dir, trg_filename)

    # Check the dataset has been extracted properly, otherwise raise an error.
    if src_path and trg_path:
        raw_files["src"].append(src_path)
        raw_files["trg"].append(trg_path)
        return raw_files
    raise OSError(f"Extraction failed for url {url} to path {raw_dir}")


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def compile_files(raw_dir, raw_files, prefix):
    '''Merge the src/trg files of two languages to concenate a raw-src/trg file'''
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath

    sys.stderr.write(f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'\
                    f'    - SRC: {src_inf}, and\n' \
                    f'    - TRG: {trg_inf}.\n')
            with open(src_inf, newline='\n') as src_inf, open(trg_inf, newline='\n') as trg_inf:
                cntr = 0 # Counter for checking
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath


def encode_file(bpe, in_file, out_file):
    sys.stderr.write(f"Read raw content from {in_file} and \n"\
            f"Write encoded content to {out_file}\n")
    
    # Use codecs lib to open langauge raw file such as 'de-en.en'
    with codecs.open(in_file, encoding='utf-8') as in_f:
        with codecs.open(out_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                # Utilize the process_line method from apply_bpe instance to encode single line.
                out_f.write(bpe.process_line(line))


def encode_files(bpe, src_in_file, trg_in_file, data_dir, prefix):
    '''Encode raw lang files with bpe'''
    src_out_file = os.path.join(data_dir, f"{prefix}.src")
    trg_out_file = os.path.join(data_dir, f"{prefix}.trg")

    if os.path.isfile(src_out_file) and os.path.isfile(trg_out_file):
        sys.stderr.write(f"Encoded files found, skip the encoding process ...\n")

    encode_file(bpe, src_in_file, src_out_file)
    encode_file(bpe, trg_in_file, trg_out_file)
    return src_out_file, trg_out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True, help='Directory to save all raw data')
    parser.add_argument('--bpe_dir', required=True, help='Directory to save file after bpe')
    parser.add_argument('--codes', required=True)
    parser.add_argument('--save_data', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--verbose', action='store_true', help='Control bpe logs printing')
    parser.add_argument('--symbols', '-s', type=int, default=32000, help="Vocabulary size")
    parser.add_argument('--min-frequency', type=int, default=6, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument('--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument('--total-symbols', '-t', action="store_true")
    opt = parser.parse_args()

    # Create folder if needed.
    mkdir_if_needed(opt.raw_dir)
    mkdir_if_needed(opt.bpe_dir)

    # Extract raw data.
    raw_train = extract_raw_files(opt.raw_dir, _TRAIN_DATA_SOURCE)
    raw_val = extract_raw_files(opt.raw_dir, _VAL_DATA_SOURCE)
    raw_test = extract_raw_files(opt.raw_dir, _TEST_DATA_SOURCE)

    # Merge files into one.
    train_src, train_trg = compile_files(opt.raw_dir, raw_train, opt.prefix + '-train')
    val_src, val_trg = compile_files(opt.raw_dir, raw_val, opt.prefix + '-val')
    test_src, test_trg = compile_files(opt.raw_dir, raw_test, opt.prefix + '-test')

    # Build up the code from training files if not exist
    opt.codes = os.path.join(opt.bpe_dir, opt.codes)
    if not os.path.isfile(opt.codes):
        sys.stderr.write(f"Collect codes from training data and save to {opt.codes}.\n")
        # Learn Byte Pair Encoding from input corpus to generate a BPE coded file.
        # A BPE coded file list the byte-pairs according to the ocurrence frequency.
        # func learn_bpe(infile_names, outfile_name, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False)
        learn_bpe(raw_train['src'] + raw_train['trg'], opt.codes, opt.symbols, opt.min_frequency, opt.verbose)
    sys.stderr.write(f"BPE codes prepared.\n")

    sys.stderr.write(f"Build up the tokenizer.\n")
    with codecs.open(opt.codes, encoding='utf-8') as codes:
        # Apply the BPE coded file to the corpus to generate encoded corpus.
        # class (apply_bpe) BPE init(codes, merges=-1, separator='@@', vocab=None, glossaries=None)
        bpe = BPE(codes, separator=opt.separator)

    sys.stderr.write(f"Encoding ...\n")
    # Pass the built apply_bpe instance to encode corpus.
    encode_files(bpe, train_src, train_trg, opt.bpe_dir, opt.prefix + '-train')
    encode_files(bpe, val_src, val_trg, opt.bpe_dir, opt.prefix + '-val')
    encode_files(bpe, test_src, test_trg, opt.bpe_dir, opt.prefix + '-test')
    sys.stderr.write(f"Done.\n")


    field = torchtext.legacy.data.Field(
        tokenize=str.split,
        lower=True,
        pad_token=Constants.PAD_WORD,
        init_token=Constants.BOS_WORD,
        eos_token=Constants.EOS_WORD)

    fields = (field, field)

    MAX_LEN = opt.max_len

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    enc_train_files_prefix = opt.prefix + '-train'
    train = TranslationDataset(
        fields=fields,
        path=os.path.join(opt.bpe_dir, enc_train_files_prefix),
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    field.build_vocab(chain(train.src, train.trg), min_freq=2)

    data = { 'settings': opt, 'vocab': field, }
    opt.save_data = os.path.join(opt.bpe_dir, opt.save_data)

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))



if __name__ == '__main__':
    main()
