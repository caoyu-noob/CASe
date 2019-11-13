import json
import os
import re
from random import choice
from tqdm import tqdm
import argparse

from newsqa_data_processing import NewsQaDataset

class BaseTransformer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

class CNNDailyTransform(BaseTransformer):
    def __init__(self, type, input_path, output_path):
        super(CNNDailyTransform, self).__init__(input_path, output_path)
        self.type = type

    def findEntityInQuestion(self, question):
        return re.findall(r"@entity\d+", question)

    def removeSpace(self, context):
        while context[0] == ' ':
            context = context[1:]
        return context

    def calculateDistanceForEntity(self, entities_indices_map, answer_index):
        distance = 0
        for indices in entities_indices_map.values():
            min_distance = abs(indices[0] - answer_index)
            for index in indices:
                if abs(index - answer_index) < min_distance:
                    min_distance = abs(index - answer_index)
            distance += min_distance
        return distance

    def findAnswerSpanInContext(self, context, question_entities, answer_indices):
        if len(answer_indices) == 0:
            return -1
        elif len(answer_indices) == 1:
            return answer_indices[0]
        else:
            if len(question_entities) == 0:
                return choice(answer_indices)
            else:
                question_entities_context_indices_map = {}
                for entity in question_entities:
                    question_entity_context_indices = [i.start() for i in re.finditer(entity + '[\s,.?!\"\')]', context)]
                    if len(question_entity_context_indices) > 0:
                        question_entities_context_indices_map[entity] = question_entity_context_indices
                min_distance = self.calculateDistanceForEntity(question_entities_context_indices_map, answer_indices[0])
                res_index = answer_indices[0]
                for i in range(1, len(answer_indices)):
                    distance = self.calculateDistanceForEntity(question_entities_context_indices_map, answer_indices[i])
                    if distance < min_distance:
                        min_distance, res_index = distance, answer_indices[i]
                return res_index

    def generateMasksEntityMap(self, data):
        mask_entity_map = {}
        for i in range(8, len(data)):
            current = data[i].split(":")
            mask, entity = current[0], current[1]
            if entity[-1] == '\n':
                entity = entity[:-1]
            mask_entity_map[mask] = entity
        return mask_entity_map

    def replaceMasksWithEntities(self, mask_entity_map, answer_start_index, context, question):
        question, _ = self.replaceMasksCore(question, mask_entity_map)
        context, new_answer_start_index = \
                self.replaceMasksCore(context, mask_entity_map, answer_start_index=answer_start_index)
        return context, question, new_answer_start_index

    def replaceMasksCore(self, context, mask_entity_map, answer_start_index=None):
        entity_iter = re.finditer(r"@entity\d+", context)
        replace_offset = 0
        res_context = "" + context
        new_answer_start_index = None
        for iter in entity_iter:
            entity, start_index = iter.group(), iter.start()
            if answer_start_index and start_index == answer_start_index:
                new_answer_start_index = start_index + replace_offset
            if start_index + len(entity) + replace_offset < len(res_context):
                res_context = res_context[:start_index + replace_offset] + mask_entity_map[entity] + \
                              res_context[start_index + len(entity) + replace_offset:]
            else:
                res_context = res_context[:start_index + replace_offset] + mask_entity_map[entity]
            replace_offset = replace_offset + len(mask_entity_map[entity]) - len(entity)
        return res_context, new_answer_start_index

    def transform(self):
        names = ['training', 'validation']
        out_names = ['train', 'dev']
        for i in range(2):
            filenames = []
            root_path = os.path.join(self.input_path, 'questions', names[i])
            for ps, ds, fs in os.walk(root_path):
                for f in fs:
                    filenames.append(os.path.join(root_path, f))

            answer_indices_num_map = {}
            answer_span = []
            total_data = []
            length = 0
            for filename in tqdm(filenames):
                with open(filename, 'r') as f:
                    id = filename[3:].split('.')[0]
                    data = f.readlines()
                    context = data[2]
                    ## we need to remove the entity for the title which is useless
                    # start_index = context.find(')', 0)
                    context = self.removeSpace(context)
                    question = data[4]
                    if question[-1] == '\n':
                        question = question[:-1]
                    question_entities = self.findEntityInQuestion(question)
                    answer = data[6]
                    if answer[-1] == '\n':
                        answer = answer[:-1]
                    answer_indices = [i.start() for i in re.finditer(answer + '[\s,.?!\"\')]', context)]
                    # print(len(answer_indices))
                    answer_num = len(answer_indices)
                    if answer_indices_num_map.__contains__(answer_num):
                        answer_indices_num_map[answer_num] = answer_indices_num_map[answer_num] + 1
                    else:
                        answer_indices_num_map[answer_num] = 1
                    answer_start_index = self.findAnswerSpanInContext(context, question_entities, answer_indices)
                    if answer_start_index >= 0:
                        answer_span.append(answer_start_index)
                        mask_entity_map = self.generateMasksEntityMap(data)
                        context, question, answer_start_index = \
                            self.replaceMasksWithEntities(mask_entity_map, answer_start_index, context, question)
                        if answer_start_index:
                            answer_entity = mask_entity_map[answer]
                            answer_list = [{'answer_start': answer_start_index, 'text': answer_entity}]
                            answer_object = {'answers': answer_list, 'question': question, 'id': id}
                            paragraph_object = {'context': context, 'qas': [answer_object]}
                            paragraphs = [paragraph_object]
                            total_data.append({'title': self.type, 'paragraphs': paragraphs})
                            length += 1
            total_data = {'data': total_data, 'version': '1.0', 'len': length}
            file_name = self.type + '_' + out_names[i] + '.json'
            with open(os.path.join(self.output_path, file_name), 'w') as f:
                json.dump(total_data, f)
            print(file_name + ' has been saved!')

class NewsQATransformer(BaseTransformer):
    def __init__(self, input_path, output_path):
        super(NewsQATransformer, self).__init__(input_path, output_path)

    def find_invalid_indices(self, context):
        start = -1
        flag = False
        remove_indices = []
        for i, c in enumerate(context):
            if c == '\n':
                if flag == False:
                    start = i
                    flag = True
            else:
                if flag == True:
                    remove_indices.append((start, i))
                    flag = False
        return remove_indices

    def remove_based_on_indices(self, context, remove_indices):
        offset = 0
        for index in remove_indices:
            context = context[:index[0] - offset] + " " + context[index[1] - offset:]
            offset += (index[1] - index[0] - 1)
        return context

    def transform_to_SQuAD(self, data):
        train_data = []
        dev_data = []
        dev_count = 0
        train_count = 0
        for d in data['data']:
            sample = {}
            sample['title'] = 'cnn'
            paragraph = {}
            context = d['text']
            split_index = context.find('--')
            offset = 0
            if split_index != -1 and split_index < 50:
                context = context[split_index + 3:]
                offset = split_index + 3
            qas = []
            remove_indices = self.find_invalid_indices(context)
            for i, qa in enumerate(d['questions']):
                if qa.__contains__('consensus') and qa['consensus'].__contains__('s') and qa['consensus'].__contains__(
                        'e'):
                    start = qa['consensus']['s'] - offset
                    answer = context[start: qa['consensus']['e'] - offset]
                    if len(answer) == 0:
                        continue
                    while answer[-1] == ' ' or answer[-1] == '\n':
                        answer = answer[:-1]
                    tmp_offset = 0
                    for index in remove_indices:
                        if index[0] - tmp_offset > start:
                            break
                        start -= (index[1] - index[0] - 1)
                        tmp_offset += (index[1] - index[0] - 1)
                    if start < 0:
                        continue
                    qas.append({'answers': [{'answer_start': start, 'text': answer}] * 3,
                                'question': qa['q'], 'id': d['storyId'] + '_' + str(i)})
            context = self.remove_based_on_indices(context, remove_indices)
            paragraph['context'] = context
            for qa in qas:
                start = qa['answers'][0]['answer_start']
                length = len(qa['answers'][0]['text'])
                if context[start:start + length] != qa['answers'][0]['text']:
                    answer = qa['answers'][0]['text']
                    invalid_indices = self.find_invalid_indices(answer)
                    answer = self.remove_based_on_indices(answer, invalid_indices)
                    length = len(answer)
                    for q in qa['answers']:
                        q['text'] = answer
                    if context[start:start + length] != answer:
                        print('wrong')
            if len(qas) != 0:
                paragraph['qas'] = qas
                sample['paragraphs'] = [paragraph]
                if d['type'] == 'train':
                    train_data.append(sample)
                    train_count += len(qas)
                elif d['type'] == 'dev':
                    dev_data.append(sample)
                    dev_count += len(qas)
        new_train_data = {'data': train_data, 'version': '1.0', 'len': train_count}
        new_dev_data = {'data': dev_data, 'version': '1.0', 'len': dev_count}
        with open(os.path.join(self.output_path, 'newsqa_dev.json'), 'w') as f:
            json.dump(new_dev_data, f)
        print('newsqa_dev.json has been saved!')
        with open(os.path.join(self.output_path, 'newsqa_train.json'), 'w') as f:
            json.dump(new_train_data, f)
        print('newsqa_train.json has been saved!')

    def transform(self):
        newsqa_data = NewsQaDataset(os.path.join(self.input_path, 'cnn_stories.tgz'),
                os.path.join(self.input_path, 'newsqa-data-v1.csv'))
        newsqa_data = newsqa_data.to_dict()
        self.transform_to_SQuAD(newsqa_data)

class CoQATransformer(BaseTransformer):
    def __init__(self, input_path, output_path):
        super(CoQATransformer, self).__init__(input_path, output_path)

    def transform(self):
        input_names = ['coqa-train-v1.0.json', 'coqa-dev-v1.0.json']
        output_names = ['coqa_train.json', 'coqa_dev.json']
        for i in range(2):
            with open(os.path.join(self.input_path, input_names[i]), 'r') as f:
                data = json.load(f)
            question_count = 0
            new_coqa_data = []
            for d in tqdm(data['data']):
                cur_data = {'title': d['filename']}
                paragraph = {}
                paragraph['context'] = d['story']
                paragraph['qas'] = []
                id = d['id']
                prev_question = ''
                for index in range(len(d['questions'])):
                    qas = {'question': prev_question + d['questions'][index]['input_text']}
                    answer = {'answer_start': d['answers'][index]['span_start'],
                              'text': d['answers'][index]['span_text']}
                    qas['answers'] = [answer] * 3
                    qas['id'] = id + str(index)
                    paragraph['qas'].append(qas)
                    prev_question = prev_question + d['questions'][index]['input_text'] + '. ' \
                                    + d['answers'][index]['input_text'] + ' '
                cur_data['paragraphs'] = [paragraph]
                question_count += len(d['questions'])
                new_coqa_data.append(cur_data)
            data = {'data': new_coqa_data, 'version': '1.0', 'len': question_count}
            with open(os.path.join(self.output_path, output_names[i]), 'w') as f:
                json.dump(data, f)
            print(output_names[i] + ' has been saved!')

class DROPTransformer(BaseTransformer):
    def __init__(self, input_path, output_path):
        super(DROPTransformer, self).__init__(input_path, output_path)

    def transform(self):
        input_names = ['drop_dataset_train.json', 'drop_dataset_dev.json']
        output_names = ['drop_train.json', 'drop_dev.json']
        for i in range(2):
            with open(os.path.join(self.input_path, input_names[i]), 'r') as f:
                data = json.load(f)
            new_data = []
            q_count = 0
            tmp_count = 0
            for k, d in tqdm(data.items()):
                sample = {}
                sample['title'] = k
                paragraph = {}
                paragraph['context'] = d['passage']
                qas = []
                for qa in d['qa_pairs']:
                    if len(qa['answer']['spans']) != 0:
                        tmp_count += 1
                        answer = qa['answer']['spans'][0]
                        answer_start = d['passage'].find(answer)
                        if answer_start != -1:
                            qas.append({'answers': [{'answer_start': answer_start, 'text': answer}] * 3,
                                        'question': qa['question'], 'id': qa['query_id']})
                if len(qas) != 0:
                    q_count += len(qas)
                    paragraph['qas'] = qas
                    sample['paragraphs'] = paragraph
                    new_data.append(sample)
            data = {'data': new_data, 'version': '1.0', 'len': q_count}
            with open(os.path.join(self.output_path, output_names[i]), 'w') as f:
                json.dump(data, f)
            print(output_names[i] + ' has been saved!')

if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_type', required=True, type=str, help='The specific source dataset type')
    parser.add_argument('--path', default=dir_name, type=str, help='The dataset path for transformation')
    parser.add_argument('--output_path', default=dir_name, type=str, help='The output path for processed file')
    all_dataset_types = ['cnn', 'dailymail', 'newsqa', 'coqa', 'drop']

    args = parser.parse_args()
    if args.dataset_type.lower() not in all_dataset_types:
        raise NameError('Cannot find the corresponding dataset type')

    if args.dataset_type.lower() == 'cnn' or args.dataset_type.lower() == 'dailymail':
        transfomer = CNNDailyTransform(args.dataset_type, args.path, args.output_path)
    elif args.dataset_type.lower() == 'newsqa':
        transfomer = NewsQATransformer(args.path, args.output_path)
    elif args.dataset_type.lower() == 'coqa':
        transfomer = CoQATransformer(args.path, args.output_path)
    elif args.dataset_type.lower() == 'drop':
        transfomer = DROPTransformer(args.path, args.output_path)
    transfomer.transform()
    print('Finished!')

