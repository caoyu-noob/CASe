# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

'''The code is based on BERT in Pytorch https://github.com/huggingface/transformers'''

import argparse
import collections
import json
import math
import os
import random
import pickle
import sys
from tqdm import tqdm, trange
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from pathlib import Path

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertForQuestionAnsweringCASe
from pytorch_pretrained_bert.optimization import BertAdam
from utils.ConfigLogger import config_logger
from utils.evaluate import f1_score, exact_match_score, metric_max_over_ground_truths
from utils.BERTRandomSampler import BERTRandomSampler
from model.network import AdversarialNetwork, RandomLayer, calc_coeff
from model.loss import CDAN, Entropy, uncon_adv

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

class SquadExample(object):
    """A single training/test example for the Squad dataset."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 answers=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.orig_answers = answers
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

class InputFeaturesSimple:
    def __init__(self, unique_id, example_index, doc_span_index, input_ids, input_mask, segment_ids, start_position,
            end_position):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

def read_squad_len(input_file):
    with open(input_file, 'r', encoding='utf-8') as reader:
        squad_len = json.load(reader)['len']
    return squad_len

def read_squad_examples(input_file, is_training, logger):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                answers = None
                if is_training:
                    # if len(qa["answers"]) != 1:
                    #     raise ValueError(
                    #         "For training, each question should have exactly 1 answer.")
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    answers = list(map(lambda x: x['text'], qa['answers']))
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                        continue

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    answers=answers)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training, logger, use_simple_feature=False):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for example_index in tqdm(range(len(examples))):
        example = examples[example_index]
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (example.start_position < doc_start or
                        example.end_position < doc_start or
                        example.start_position > doc_end or example.end_position > doc_end):
                    continue

                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))
            if use_simple_feature:
                features.append(InputFeaturesSimple(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position))
            else:
                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index



RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging, logger, write_json):
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_probs = []
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        all_probs.append(0)
        if len(features) == 0:
            continue

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_probs[example_index] = probs[0]
        all_nbest_json[example.qas_id] = nbest_json

    if write_json:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    return all_predictions, all_probs


def get_final_text(pred_text, orig_text, do_lower_case, logger, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def prediction_stage(args, device, tokenizer, logger, debug=False):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.output_dir, args.output_model_file)
    model_state_dict = torch.load(output_model_file)
    model = BertForQuestionAnswering.from_pretrained(args.bert_model, state_dict=model_state_dict, args=args)
    model.to(device)
    # Read prediction samples
    read_limit = None
    if debug:
        read_limit = 200
    logger.info("***** Reading Prediction Samples *****")
    eval_features, eval_examples = read_features_and_examples(args, args.predict_file, tokenizer, logger,
            use_simple_feature=False, read_examples=True, limit=read_limit)
    acc, f1 = evaluation_stage(args, eval_examples, eval_features, device, model, logger)
    logger.info('***** Prediction Performance *****')
    logger.info('EM is %.5f, F1 is %.5f', acc, f1)

def evaluate_acc_and_f1(predictions, raw_data, logger, threshold=-1, all_probs=None):
    f1 = exact_match = total = 0
    eval_threshold = True
    if threshold is None or all_probs is None:
        eval_threshold = False
    for sample in raw_data:
        if (sample.qas_id not in predictions) or (eval_threshold and sample.qas_id not in all_probs):
            message = 'Unanswered question ' + sample.qas_id + ' will receive score 0.'
            logger.warn(message)
            continue
        if not eval_threshold or (eval_threshold and all_probs[sample.qas_id] >= threshold):
            ground_truths = sample.orig_answers
            prediction = predictions[sample.qas_id]
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)
            total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1

def read_features_and_examples(args, file_name, tokenizer, logger, use_simple_feature=True, read_examples=False,
        limit=None):
    cached_features_file = file_name + '_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length))
    if use_simple_feature:
        cached_features_file = cached_features_file + '_simple'

    examples, features = None, None
    if read_examples:
        examples = read_squad_examples(input_file=file_name, is_training=True, logger=logger)
    try:
        with open(cached_features_file, "rb") as reader:
            features = pickle.load(reader)
    except:
        if examples is None:
            examples = read_squad_examples(input_file=file_name, is_training=True, logger=logger)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True, logger=logger, use_simple_feature=use_simple_feature)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving eval features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as writer:
                pickle.dump(features, writer)
    if limit is not None:
        features = features[:limit]
        if examples is not None:
            examples = examples[:limit]
    return features, examples

def keep_high_prob_samples(all_probs, all_features, prob_threshold, removed_feature_index, keep_generated=False):
    new_train_features = []
    for feature in all_features:
        if keep_generated:
            if feature.example_index not in removed_feature_index and all_probs[feature.example_index] > prob_threshold:
                new_train_features.append(feature)
                removed_feature_index.add(feature.example_index)
        else:
            if all_probs[feature.example_index] > prob_threshold:
                new_train_features.append(feature)
    return new_train_features, removed_feature_index

def compare_performance(args, best_acc, best_f1, acc, f1, model, logger):
    if not (best_f1 is None or best_acc is None):
        if best_acc < acc:
            logger.info('Current model BEATS previous best model, previous best is EM = %.5F, F1 = %.5f',
                best_acc, best_f1)
            best_acc, best_f1 = acc, f1
            logger.info('Current best model has been saved!')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, args.output_model_file))
        else:
            logger.info('Current model CANNOT beat previous best model, previous best is EM = %.5F, F1 = %.5f',
                best_acc, best_f1)
    else:
        best_acc, best_f1 = acc, f1
    return best_acc, best_f1

def evaluation_stage(args, eval_examples, eval_features, device, model, logger, generate_prob_th=0.7,
        removed_feature_index=None, global_step=None, best_acc=None, best_f1=None, generate_label=False):
    if not global_step:
        logger.info("***** Running Evaluation Stage *****")
    else:
        logger.info("***** Running Predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
             batch_start_logits, batch_end_logits, _ = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    if global_step:
        prediction_file_name = 'predictions_' + str(global_step) + '.json'
        nbest_file_name = 'nbest_predictions_' + str(global_step) + '.json'
        output_prediction_file = os.path.join(args.output_dir, prediction_file_name)
        output_nbest_file = os.path.join(args.output_dir, nbest_file_name)
    else:
        output_prediction_file = os.path.join(args.output_dir, 'predictions.json')
        output_nbest_file = os.path.join(args.output_dir, 'nbest_predictions.json')
    all_predictions, all_probs = write_predictions(eval_examples, eval_features, all_results,
        args.n_best_size, args.max_answer_length,
        args.do_lower_case, output_prediction_file,
        output_nbest_file, args.verbose_logging, logger, args.output_prediction)
    if generate_label:
        return keep_high_prob_samples(all_probs, eval_features, generate_prob_th, removed_feature_index,
                keep_generated=args.keep_previous_generated)
    else:
        acc, f1 = evaluate_acc_and_f1(all_predictions, eval_examples, logger)
        logger.info('Current EM is %.5f, F1 is %.5f', acc, f1)
        if not (best_f1 is None or best_acc is None):
            best_acc, best_f1 = compare_performance(args, best_acc, best_f1, acc, f1, model, logger)
            return best_acc, best_f1
        else:
            return acc, f1

def generate_self_training_samples(args, train_examples, train_features, device, model, removed_feature_index,
        new_generated_train_features, generate_prob_th, logger):
    logger.info('***** Generating training data for this epoch *****')
    if args.keep_previous_generated:
        train_features_removed_previous = []
        for index in range(len(train_features)):
            if index not in removed_feature_index:
                train_features_removed_previous.append(train_features[index])
    else:
        train_features_removed_previous = train_features
    cur_train_features, removed_feature_index = \
        evaluation_stage(args, train_examples, train_features_removed_previous, device, model, logger,
            removed_feature_index=removed_feature_index, generate_label=True, generate_prob_th=generate_prob_th)
    new_generated_train_features = cur_train_features
    if len(cur_train_features) == 0:
        logger.info("  No new training samples were generated, training procedure ends")
        return None, None
    if args.keep_previous_generated:
        new_generated_train_features.extend(cur_train_features)
    else:
        new_generated_train_features = cur_train_features
    return new_generated_train_features, removed_feature_index

def labeled_training_loss(start_logits, end_logits, start_positions, end_positions):
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    loss = (start_loss + end_loss) / 2
    return loss

def get_bert_model_parameters(model):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters

def self_training_stage(args, train_examples, train_features, eval_examples, eval_features, device, model,
        removed_feature_index, new_generated_train_features, generate_prob_th, n_gpu, lr_decay_rate, epoch, best_acc,
        best_f1, logger):
    logger.info('\n')
    logger.info('====================  Start Self Training Stage  ====================')
    new_generated_train_features, removed_feature_index = generate_self_training_samples(args, train_examples,
            train_features, device, model, removed_feature_index, new_generated_train_features, generate_prob_th,
            logger)
    if new_generated_train_features is None:
        sys.exit()
    logger.info("  Num split examples = %d", len(new_generated_train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    num_train_steps = int(
        len(new_generated_train_features) / args.train_batch_size / args.gradient_accumulation_steps)
    if num_train_steps == 0 and len(new_generated_train_features) > 0:
        num_train_steps = 1
    logger.info("  Num steps = %d", num_train_steps)

    global_step = 0
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer_grouped_parameters = get_bert_model_parameters(model)
    # Prepare Optimizer for model
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
            lr=args.self_learning_rate,
            bias_correction=False,
            max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
            lr=args.self_learning_rate,
            warmup=args.warmup_proportion,
            t_total=t_total)

    all_input_ids = torch.tensor([f.input_ids for f in new_generated_train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in new_generated_train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in new_generated_train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in new_generated_train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in new_generated_train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
        all_start_positions, all_end_positions)
    if args.local_rank == -1:
        train_sampler = BERTRandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    # train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    loss_sum = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        model.train()
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.self_learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion) \
                           * lr_decay_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            # print(optimizer.param_groups[0]['params'][75])
            optimizer.zero_grad()
            global_step += 1
        loss_sum += loss.data.cpu().numpy()
        if global_step % args.loss_logging_interval == 0:
            logger.info('Current loss is %.3f', loss_sum / args.loss_logging_interval)
            loss_sum = 0
        if global_step % args.evaluation_interval == 0:
            best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
                global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
    final_acc, final_f1 = None, None
    if epoch == args.num_train_epochs - 1:
        final_acc, final_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
            global_step=global_step, best_acc=None, best_f1=None, logger=logger)
        best_acc, best_f1 = compare_performance(args, best_acc, best_f1, final_acc, final_f1, model, logger)
    else:
        best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
            global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
    return best_acc, best_f1, final_acc, final_f1

def calculate_loss_for_bert(start_logits, end_logits, start_positions, end_positions):
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

def comb_adversarial_training_stage(args, target_train_features, target_train_examples, source_train_features,
        eval_features, eval_examples, removed_feature_index, new_generated_train_features, model, adv_net, random_layer,
        epoch, n_gpu, device, best_acc, best_f1, logger):
    ## Generate self-training samples
    new_generated_train_features, removed_feature_index = generate_self_training_samples(args, target_train_examples,
        target_train_features, device, model, removed_feature_index, new_generated_train_features, args.generate_prob_th,
        logger)
    if new_generated_train_features is None:
        sys.exit()
    logger.info('\n')
    logger.info('====================  Start Adversarial Training Stage  ====================')
    target_input_ids = torch.tensor([f.input_ids for f in new_generated_train_features], dtype=torch.long)
    target_input_mask = torch.tensor([f.input_mask for f in new_generated_train_features], dtype=torch.long)
    target_segment_ids = torch.tensor([f.segment_ids for f in new_generated_train_features], dtype=torch.long)
    target_start_positions = torch.tensor([f.start_position for f in new_generated_train_features], dtype=torch.long)
    target_end_positions = torch.tensor([f.end_position for f in new_generated_train_features], dtype=torch.long)
    source_input_ids = torch.tensor([f.input_ids for f in source_train_features], dtype=torch.long)
    source_input_mask = torch.tensor([f.input_mask for f in source_train_features], dtype=torch.long)
    source_segment_ids = torch.tensor([f.segment_ids for f in source_train_features], dtype=torch.long)
    target_data = TensorDataset(target_input_ids, target_input_mask, target_segment_ids, target_start_positions,
            target_end_positions)
    source_data = TensorDataset(source_input_ids, source_input_mask, source_segment_ids)
    if args.local_rank == -1:
        target_sampler = BERTRandomSampler(target_data)
        source_sampler = BERTRandomSampler(source_data)
    else:
        target_sampler = DistributedSampler(target_data)
        source_sampler = DistributedSampler(source_data)
    target_dataloader = DataLoader(target_data, sampler=target_sampler, batch_size=int(args.train_batch_size / 2))
    source_dataloader = DataLoader(source_data, sampler=source_sampler, batch_size=int(args.train_batch_size / 2))
    data_len = min(len(source_train_features), len(target_train_features))
    logger.info("  Num split examples = %d", data_len)
    logger.info("  Batch size = %d", args.train_batch_size)
    num_train_steps = int(data_len / (args.train_batch_size / 2) / args.gradient_accumulation_steps)
    if num_train_steps == 0 and data_len > 0:
        num_train_steps = 1
    t_total = num_train_steps
    logger.info("  Num steps = %d", num_train_steps)

    loss_sum = 0
    optimizer_grouped_parameters = get_bert_model_parameters(model)
    optimizer_grouped_parameters[0]['params']  = optimizer_grouped_parameters[0]['params'] + list(adv_net.parameters())
    optimizer = BertAdam(optimizer_grouped_parameters,
            lr=args.self_learning_rate,
            warmup=args.warmup_proportion,
            t_total=t_total)
    optimizer_ad = torch.optim.Adam(adv_net.parameters(), lr=args.adv_learning_rate, weight_decay=0.0005)
    global_step = 0
    softmax = torch.nn.Softmax(dim=-1)
    for step, batch_target in enumerate(tqdm(target_dataloader, desc="Iteration")):
        try:
            batch_source = source_dataloader.__iter__().__next__()
        except:
            logger.info("  All data in source dataset has been used  ")
            break
        if n_gpu == 1:
            batch_target = tuple(t.to(device) for t in batch_target)
            batch_source = tuple(t.to(device) for t in batch_source)
        input_ids_target, input_masks_target, segment_ids_target, start_positions, end_positions = batch_target
        input_ids_source, input_masks_source, segment_ids_source = batch_source
        batch_size_target, batch_size_source = input_ids_target.shape[0], input_ids_source.shape[0]
        input_ids = torch.cat((input_ids_target, input_ids_source), 0)
        input_masks = torch.cat((input_masks_target, input_masks_source), 0)
        segment_ids = torch.cat((segment_ids_target, segment_ids_source), 0)
        start_logits, end_logits, sequence_out, loss = model(input_ids, input_masks, segment_ids, start_positions,
                end_positions, combine_train=True)
        # loss = calculate_loss_for_bert(start_logits[:batch_size_target], end_logits[:batch_size_target],
        #         start_positions, end_positions)

        start_logits = softmax(start_logits)
        end_logits = softmax(end_logits)
        if args.CASe_method == 'CASe':
            transfer_loss = CDAN([sequence_out, start_logits, end_logits], adv_net,
                    [batch_size_target, batch_size_source], None, None, random_layer)
            if args.adv_method == 'unconditional':
                transfer_loss = uncon_adv(sequence_out, adv_net, [batch_size_target, batch_size_source])
        else:
            entropy = Entropy(torch.cat((start_logits, end_logits), -1))
            coeff = calc_coeff(num_train_steps * epoch + step)
            transfer_loss = CDAN([sequence_out, start_logits, end_logits], adv_net,
                    [batch_size_target, batch_size_source], entropy, coeff, random_layer)
        if n_gpu > 1:
            loss = loss.mean()
        loss += transfer_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_ad.step()
        optimizer_ad.zero_grad()
        global_step += 1
        loss_sum += loss.data.cpu().numpy()
        if global_step % (args.loss_logging_interval) == 0:
            logger.info('Current loss is %.3f', loss_sum / (args.loss_logging_interval * 5))
            loss_sum = 0
        if global_step % (args.evaluation_interval) == 0:
            best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
                global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
    best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
        global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
    return best_acc, best_f1

def adversarial_training_stage(args, target_train_features, source_train_features, eval_features, eval_examples, model,
        adv_net, random_layer, epoch, n_gpu, device, best_acc, best_f1, logger):
    logger.info('\n')
    logger.info('====================  Start Adversarial Training Stage  ====================')
    target_input_ids = torch.tensor([f.input_ids for f in target_train_features], dtype=torch.long)
    target_input_mask = torch.tensor([f.input_mask for f in target_train_features], dtype=torch.long)
    target_segment_ids = torch.tensor([f.segment_ids for f in target_train_features], dtype=torch.long)
    source_input_ids = torch.tensor([f.input_ids for f in source_train_features], dtype=torch.long)
    source_input_mask = torch.tensor([f.input_mask for f in source_train_features], dtype=torch.long)
    source_segment_ids = torch.tensor([f.segment_ids for f in source_train_features], dtype=torch.long)
    target_data = TensorDataset(target_input_ids, target_input_mask, target_segment_ids)
    source_data = TensorDataset(source_input_ids, source_input_mask, source_segment_ids)
    if args.local_rank == -1:
        target_sampler = BERTRandomSampler(target_data)
        source_sampler = BERTRandomSampler(source_data)
    else:
        target_sampler = DistributedSampler(target_data)
        source_sampler = DistributedSampler(source_data)
    if (len(target_train_features) > len(source_train_features)):
        adv_dataloader_smaller = DataLoader(source_data, sampler=source_sampler,
                batch_size=int(args.train_batch_size / 2))
        adv_dataloader_larger = DataLoader(target_data, sampler=target_sampler,
                batch_size=int(args.train_batch_size / 2))
        logger.info("  Num split examples = %d", len(source_train_features))
        data_len = len(source_train_features)
    else:
        adv_dataloader_larger = DataLoader(source_data, sampler=source_sampler,
                batch_size=int(args.train_batch_size / 2))
        adv_dataloader_smaller = DataLoader(target_data, sampler=target_sampler,
                batch_size=int(args.train_batch_size / 2))
        logger.info("  Num split examples = %d", len(target_train_features))
        data_len = len(target_train_features)
    logger.info("  Batch size = %d", args.train_batch_size)
    num_train_steps = int(data_len / (args.train_batch_size / 2) / args.gradient_accumulation_steps)
    if num_train_steps == 0 and data_len > 0:
        num_train_steps = 1
    t_total = num_train_steps
    logger.info("  Num steps = %d", num_train_steps)

    loss_sum = 0
    optimizer_grouped_parameters = get_bert_model_parameters(model)
    optimizer_grouped_parameters[0]['params']  = optimizer_grouped_parameters[0]['params'] + list(adv_net.parameters())
    optimizer = BertAdam(optimizer_grouped_parameters,
            lr=args.adv_learning_rate,
            warmup=args.warmup_proportion,
            t_total=t_total)
    global_step = 0
    softmax = torch.nn.Softmax(dim=-1)
    for step, batch_smaller in enumerate(tqdm(adv_dataloader_smaller, desc="Iteration")):
        batch_larger = adv_dataloader_larger.__iter__().__next__()
        if n_gpu == 1:
            batch_smaller = tuple(t.to(device) for t in batch_smaller)
            batch_larger = tuple(t.to(device) for t in batch_larger)
        input_ids_smaller, input_masks_smaller, segment_ids_smaller = batch_smaller
        input_ids_larger, input_masks_larger, segment_ids_larger = batch_larger
        batch_size_larger, batch_size_smaller = input_ids_larger.shape[0], input_ids_smaller.shape[0]
        input_ids = torch.cat((input_ids_smaller, input_ids_larger), 0)
        input_masks = torch.cat((input_masks_smaller, input_masks_larger), 0)
        segment_ids = torch.cat((segment_ids_smaller, segment_ids_larger), 0)
        start_logits, end_logits, sequence_out = model(input_ids, input_masks, segment_ids)
        start_logits = softmax(start_logits)
        end_logits = softmax(end_logits)
        if args.CASe_method == 'CASe':
            transfer_loss = CDAN([sequence_out, start_logits, end_logits], adv_net,
                    [batch_size_larger, batch_size_smaller], None, None, random_layer)
        else:
            entropy = Entropy(torch.cat((start_logits, end_logits), -1))
            coeff = calc_coeff(num_train_steps * epoch + step)
            transfer_loss = CDAN([sequence_out, start_logits, end_logits], adv_net,
                    [batch_size_larger, batch_size_smaller], entropy, coeff, random_layer)
        transfer_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        loss_sum += transfer_loss.data.cpu().numpy()
        if global_step % (args.loss_logging_interval * 5) == 0:
            logger.info('Current loss is %.3f', loss_sum / (args.loss_logging_interval * 5))
            loss_sum = 0
        if global_step % (args.evaluation_interval * 5) == 0:
            best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
                global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
    best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
        global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
    return best_acc, best_f1

def prepare_model(args, device, n_gpu):
    model, adv_net, random_layer = None, None, None
    if args.do_train:
        model = BertForQuestionAnswering.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), args=args)
        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
    elif args.do_adaptation:
        input_model_file = os.path.join(args.input_dir, args.input_model_file)
        model_state_dict = torch.load(input_model_file)
        model = BertForQuestionAnsweringCASe.from_pretrained(args.bert_model, state_dict=model_state_dict, args=args)
        adv_net = AdversarialNetwork(768, args.max_seq_length)
        random_layer = RandomLayer(n_gpu, device, [768, args.max_seq_length * 2], 768)
        if args.fp16:
            model.half()
            adv_net.half()
            random_layer.half()
        model.to(device)
        adv_net.to(device)
        random_layer.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
            adv_net = DDP(adv_net)
            random_layer = DDP(random_layer)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
            adv_net = torch.nn.DataParallel(adv_net)
            random_layer = torch.nn.DataParallel(random_layer)
    return model, adv_net, random_layer

def adaptation_stage(args, tokenizer, n_gpu, device, logger, debug=False):
    model, adv_net, random_layer = prepare_model(args, device, n_gpu)
    best_acc, best_f1 = 0, 0

    read_limit = None
    if debug:
        read_limit = 50

    ## Read target training examples
    logger.info("***** Reading Target Unlabeled Training Samples *****")
    train_features, train_examples = read_features_and_examples(args, args.target_train_file, tokenizer, logger,
        use_simple_feature=False, read_examples=True, limit=read_limit)

    ## Read source training examples
    logger.info("***** Reading Source Training Samples *****")
    source_train_features, _ = read_features_and_examples(args, args.source_train_file, tokenizer, logger,
        use_simple_feature=True, read_examples=True, limit=read_limit)

    # Read evaluation samples
    logger.info("***** Reading Evaluation Samples *****")
    eval_features, eval_examples = read_features_and_examples(args, args.target_predict_file, tokenizer, logger,
        use_simple_feature=False, read_examples=True, limit=read_limit)

    removed_feature_index = set()
    new_generated_train_features = []
    lr_decay_rate = 1
    final_acc, final_f1 = 0.0, 0.0
    if args.training_method == "alternated":
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info('\n')
            logger.info(' ###########  Start Adaptation Epoch %d  ###########', epoch + 1)
            logger.info('\n')
            best_acc, best_f1, final_acc, final_f1 = self_training_stage(args, train_examples, train_features,
                        eval_examples, eval_features, device, model, removed_feature_index, new_generated_train_features,
                        args.generate_prob_th, n_gpu, lr_decay_rate, epoch, best_acc, best_f1, logger)
            if epoch != args.num_train_epochs - 1:
                best_acc, best_f1 = adversarial_training_stage(args, train_features, source_train_features,
                        eval_features, eval_examples, model, adv_net, random_layer, epoch, n_gpu, device, best_acc,
                        best_f1, logger)
            logger.info('\n')
            logger.info(' ###########  End Training Epoch %d  ###########', epoch + 1)
            logger.info('\n')
    else:
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info('\n')
            logger.info(' ###########  Start Training Epoch %d  ###########', epoch + 1)
            logger.info('\n')
            best_acc, best_f1 = comb_adversarial_training_stage(args, train_features, train_examples,
                    source_train_features, eval_features, eval_examples, removed_feature_index,
                    new_generated_train_features, model, adv_net, random_layer, epoch, n_gpu, device, best_acc, best_f1,
                    logger)
            logger.info('\n')
            logger.info(' ###########  End Training Epoch %d  ###########', epoch + 1)
            logger.info('\n')

    # Save the final trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, args.output_model_file + '.final')
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)
    logger.info('The final model has been save')
    logger.info('*** The Training Stage is Ended ***')
    logger.info('\n\nBest EM is %.5f. Best F1 is %.5f', best_acc, best_f1)
    logger.info('\n\nFinal EM is %.5f. Best F1 is %.5f', final_acc, final_f1)

def training_stage(args, tokenizer, n_gpu, device, logger, debug=False):
    model, _, _ = prepare_model(args, device, n_gpu)
    read_limit = None
    if debug:
        read_limit = 50

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    train_examples_len = read_squad_len(args.train_file)
    num_train_steps = math.ceil(
        train_examples_len / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    global_step = 0
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    best_acc, best_f1 = 0, 0
    use_simple_feature = True
    if args.use_simple_feature == 0:
        use_simple_feature = False
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
            lr=args.train_learning_rate,
            bias_correction=False,
            max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
            lr=args.train_learning_rate,
            warmup=args.warmup_proportion,
            t_total=t_total)

    ## Read training examples
    logger.info("***** Reading Training Samples *****")
    train_features, _ = read_features_and_examples(args, args.train_file, tokenizer, logger,
        use_simple_feature=use_simple_feature, limit=read_limit)

    # Read evaluation samples
    logger.info("***** Reading Evaluation Samples *****")
    eval_features, eval_examples = read_features_and_examples(args, args.predict_file, tokenizer, logger,
        use_simple_feature=False, read_examples=True, limit=read_limit)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", train_examples_len)
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
        all_start_positions, all_end_positions)
    if args.local_rank == -1:
        train_sampler = BERTRandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        loss_sum = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.train_learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            loss_sum += loss.data.cpu().numpy()
            if global_step % args.loss_logging_interval == 0:
                logger.info('Current loss is %.3f', loss_sum / args.loss_logging_interval)
                loss_sum = 0
            if global_step % args.evaluation_interval == 0:
                best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
                    global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
        best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
            global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)

    # Save the final trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, args.output_model, '.final')
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)
    logger.info('The final model has been save')
    logger.info('*** The Training Stage is Ended ***')
    logger.info('\n\nBest EM is %.5f. Best F1 is %.5f', best_acc, best_f1)


def main(debug=False):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--output_model_file", default=None, type=str, required=True,
        help="The model file which will be saved after training, it cannot be empty.")

    ## Other parameters
    parser.add_argument("--input_dir", default=None, type=str,
                        help="The output directory where the pretrained model will be loaded.")
    parser.add_argument("--input_model_file", default=None, type=str,
        help="The model file which will be loaded before training, it cannot be empty.")
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions/Dev. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--target_train_file", default=None, type=str, help="Train file in target domain")
    parser.add_argument("--target_predict_file", default=None, type=str, help="Dev file in target domain")
    parser.add_argument("--source_train_file", default=None, type=str, help="Train file in source domain")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=40, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run supervised training.")
    parser.add_argument("--do_adaptation", action='store_true', help="Whether to run unsupervised domain adaptation.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--keep_previous_generated", action='store_true', help="Whether to keep the generated"+
            "samples in previous epochs, if not every epoch it will generate new samples from whole target domain")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=12, type=int, help="Total batch size for predictions.")
    parser.add_argument("--evaluation_interval", default=2000, type=int, help="Batch interval to run evaluation.")
    parser.add_argument("--loss_logging_interval", default=500, type=int, help="Batch interval to run evaluation.")
    parser.add_argument("--train_learning_rate", default=3e-5, type=float,
        help="The initial learning rate for Adam in supervised training.")
    parser.add_argument("--self_learning_rate", default=2e-5, type=float,
        help="The initial learning rate for Adam in self-training in adaptation.")
    parser.add_argument("--adv_learning_rate", default=1e-5, type=float,
        help="The initial learning rate for adversarial training")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--logger_path', type=str, default='bert', help='The path to save log of current program')
    parser.add_argument('--use_simple_feature', type=int, default=1,
                        help='Whether to use the feature of simplified version, >=1 means use simple feature')
    parser.add_argument('--CASe_method', type=str, default="CASe",
        help='The loss method to used in training, CASe or CASe+E')
    parser.add_argument('--generate_prob_th', type=float, default=0.4,
        help='The probability threshold for generating training samples in slef-training')
    parser.add_argument("--training_method", type=str, default="alternated", help="alternated=run self-traing and " +
            "adverssarial learning alernatively, comb=combine self-training with adversarial laerning")
    parser.add_argument("--adv_method", type=str, default="conditional", help="conditional adversarial training or " +
            "uncondiional adversarial training which directly put feature from BERT into adversarila network")
    parser.add_argument("--use_BN", action='store_true', help="Whether to use Batch Normalization in the output layer.")
    parser.add_argument("--output_prediction", action='store_true', help="Whether to output the prediction json file.")

    args = parser.parse_args()
    logger = config_logger(args.logger_path)
    for item in vars(args).items():
        logger.info('%s : %s', item[0], str(item[1]))

    if debug:
        args.loss_logging_interval = 10
        args.evaluation_interval = 50

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict and not args.do_adaptation:
        raise ValueError("At least one of `do_train` or `do_predict` or 'do_adaptation' must be True.")

    if args.do_train and args.do_adaptation:
        raise ValueError("Only one of `do_train` or `do_adaptation` can be True.")

    if args.do_train:
        if not args.train_file or not args.predict_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` and 'predict_file' must be specified.")

    if args.do_adaptation:
        if not args.source_train_file or not args.target_predict_file or not args.target_train_file:
            raise ValueError(
                "If `do_adaptation` is True, then `source_train_file` , 'target_train_file' "
                "and 'target_predict_file' must be specified.")
        if not args.input_dir or not args.input_model_file:
            raise ValueError("If `do_adaptation` is True, then `input_dir` and `input_model_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if (not os.path.exists(args.output_dir)) and (args.do_train or args.do_adaptation):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.do_train:
        training_stage(args, tokenizer, n_gpu, device, logger, debug=debug)

    if args.do_adaptation:
        adaptation_stage(args, tokenizer, n_gpu, device, logger, debug=debug)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and (not args.do_train):
        prediction_stage(args, device, tokenizer, logger, debug=debug)

if __name__ == "__main__":
    main(debug=False)
