import json
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('ratio', type=str)
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        data = json.load(f)
    length = data['len']
    version = data['version']
    data = data['data']
    doc_para_qas_indices = []
    for d_i, d in enumerate(data):
        for p_i, para in enumerate(d['paragraphs']):
            for q_i, qa in enumerate(para['qas']):
                doc_para_qas_indices.append((d_i, p_i, q_i))
    array = np.arange(length)
    new_data = []
    prev_d_i, prev_p_i = 0, 0
    cur_doc = {'title': data[0]['title'], 'paragraphs':[]}
    doc_para_qas_map = {}
    for i in range(0, length, int(args.ratio)):
        if not doc_para_qas_map.__contains__(doc_para_qas_indices[i][0]):
            doc_para_qas_map[doc_para_qas_indices[i][0]] = {}
        if not doc_para_qas_map[doc_para_qas_indices[i][0]].__contains__(doc_para_qas_indices[i][1]):
            doc_para_qas_map[doc_para_qas_indices[i][0]][doc_para_qas_indices[i][1]] = []
        doc_para_qas_map[doc_para_qas_indices[i][0]][doc_para_qas_indices[i][1]].append(doc_para_qas_indices[i][2])
    count = 0
    for d_i, doc in doc_para_qas_map.items():
        cur_doc = {'title': data[d_i]['title'], 'paragraphs':[]}
        for p_i, para in doc.items():
            cur_para = {'context': data[d_i]['paragraphs'][p_i]['context'], 'qas': []}
            for q_i in para:
                cur_para['qas'].append(data[d_i]['paragraphs'][p_i]['qas'][q_i])
                count += 1
            cur_doc['paragraphs'].append(cur_para)
        new_data.append(cur_doc)
    nd = {'data': new_data, 'version': version, 'len': count}
    with open(args.out_file, 'w') as f:
        json.dump(nd, f)
    print('Finished')