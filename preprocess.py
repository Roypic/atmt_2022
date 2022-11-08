import argparse
import collections
import logging
import os
import sys
import re
import pickle

# establish link to seq2seq dir
# scripts_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = os.path.join(scripts_dir, "..")
# sys.path.append(base_dir)

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default='en', metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default='fr', metavar='TGT', help='target language')

    parser.add_argument('--train-prefix', default='.//data//en-fr//preprocessed//train.bpe', metavar='FP', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default='.//data//en-fr//preprocessed//tiny-train', metavar='FP', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default='.//data//en-fr//preprocessed//valid.bpe', metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default='.//data//en-fr//preprocessed//test.bpe', metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin-dropout', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold-src', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num_words_src', default=2500, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num_words_tgt', default=2500, type=int, help='number of target words to retain')
    parser.add_argument('--vocab-src', default=None, type=str, help='path to dictionary')
    parser.add_argument('--vocab-trg', default=None, type=str, help='path to dictionary')
    parser.add_argument('--quiet', action='store_true', help='no logging')


    return parser.parse_args()


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)

    src_dict = build_dictionary([args.train_prefix + '.' + args.source_lang])

    src_dict.finalize(threshold=args.threshold_src, num_words=4000)
    src_dict.save(os.path.join(args.dest_dir, 'dict.' + args.source_lang))
    if not args.quiet:
        logging.info('Built a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))




    tgt_dict = build_dictionary([args.train_prefix + '.' + args.target_lang])

    tgt_dict.finalize(threshold=args.threshold_tgt, num_words=4000)
    tgt_dict.save(os.path.join(args.dest_dir, 'dict.' + args.target_lang))
    if not args.quiet:
        logging.info('Built a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))



    def make_split_datasets(lang, dictionary):
        if args.train_prefix is not None:
            make_binary_dataset(args.train_prefix + '.' + lang, os.path.join(args.dest_dir, 'train.' + lang),
                                dictionary)
        # if args.tiny_train_prefix is not None:
        #     make_binary_dataset(args.tiny_train_prefix + '.' + lang, os.path.join(args.dest_dir, 'tiny_train.' + lang),
        #                         dictionary)
        if args.valid_prefix is not None:
            make_binary_dataset(args.valid_prefix + '.' + lang, os.path.join(args.dest_dir, 'valid.' + lang),
                                dictionary)
        if args.test_prefix is not None:
            make_binary_dataset(args.test_prefix + '.' + lang, os.path.join(args.dest_dir, 'test.' + lang), dictionary)

    make_split_datasets(args.source_lang, src_dict)
    make_split_datasets(args.target_lang, tgt_dict)


def build_dictionary(filenames, tokenize=word_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r',encoding='utf-8') as file:
            for line in file:
                for symbol in word_tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r',encoding='utf-8') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not args.quiet:
            logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    if not args.quiet:
        utils.init_logging(args)
        logging.info('COMMAND: %s' % ' '.join(sys.argv))
        logging.info('Arguments: {}'.format(vars(args)))
    main(args)
