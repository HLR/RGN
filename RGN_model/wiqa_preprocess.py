# coding=utf-8
import logging
import jsonlines
import os
from tqdm import tqdm
import re
import copy
import json
import random

from transformers.data.processors.utils import DataProcessor, InputExample

logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label, example_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.example_id = example_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TripletInputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids,
                 aug_one_input_ids, aug_one_attention_mask, aug_one_token_type_ids,
                 aug_two_input_ids, aug_two_attention_mask, aug_two_token_type_ids,
                 label, example_id,
                 labels_one_hot, aug_labels_one_hot,
                 paired, triplet):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.aug_one_input_ids = aug_one_input_ids
        self.aug_one_attention_mask = aug_one_attention_mask
        self.aug_one_token_type_ids = aug_one_token_type_ids
        self.aug_two_input_ids = aug_two_input_ids
        self.aug_two_attention_mask = aug_two_attention_mask
        self.aug_two_token_type_ids = aug_two_token_type_ids
        self.label = label
        self.example_id = example_id
        self.labels_one_hot = labels_one_hot
        self.aug_labels_one_hot = aug_labels_one_hot
        self.paired = paired
        self.triplet = paired

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def multi_qa_convert_examples_to_features(examples, tokenizer,
                                          max_length=512,
                                          task=None,
                                          label_list=None,
                                          output_mode=None,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True):
    if task is not None:
        processor = multi_qa_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = multi_qa_output_modes[task]
            logger.info("Using output mode %s for task %s" %
                        (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        guid = example.guid
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length
        )
        # input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        input_ids, token_type_ids = inputs["input_ids"], inputs["attention_mask"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] *
                              padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" %
                        " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" %
                        " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label,
                          example_id=guid))

    return features


class WIQAProcessor(DataProcessor):
    """Processor for the WIQA data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return ["more", "less", "no_effect"]

    def _read_jsonl(self, jsonl_file):
        lines = []
        print("loading examples from {0}".format(jsonl_file))
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines

    def _create_examples(self, lines, set_type, add_consistency=True):
        """Creates examples for the training and dev sets."""
        examples = []

        for (_, data_raw) in tqdm(enumerate(lines)):
            question = data_raw["question"]["stem"]
            para_steps = " ".join(data_raw["question"]['para_steps'])
            answer_labels = data_raw["question"]["answer_label"]
            example_id = data_raw['metadata']['ques_id']
            examples.append(
                InputExample(
                    guid=example_id,
                    text_a=question,
                    text_b=para_steps,
                    label=answer_labels))
        return examples


multi_qa_tasks_num_labels = {
    "wiqa": 3,
}

multi_qa_processors = {
    "wiqa": WIQAProcessor,
}

multi_qa_output_modes = {
    "wiqa": "classification",
}
