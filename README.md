# CASe
Implementation of AAAI2020 paper " Unsupervised Domain Adaptation on Reading Comprehension "


## Requirements
1. Python 3.6
2. Pytorch >= 1.0.0

Maybe other libraries are needed, please check when you try to run.

## How to run
#### 1. Get Raw Datasets

Totally 6 datasets are included in this paper:

1. SQuAD-1.1: [Training set](https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/train-v1.1.json), [Dev Set](https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json)
2. CNN: [The whole dataset](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTTljRDVZMFJnVWM)
3. DailyMail: [The whole dataset](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfN0xhTDVteGQ3eG8)
4. NewsQA: [CSV file](https://msropendata.com/datasets/939b1042-6402-4697-9c15-7a28de7e1321)
5. CoQA: [Training set](http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json) [Dev set](http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json)
6. DROP: [The whole dataset](https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip)

#### 2. Transform all datasets into SQuAD format json (text-span based)

We need to obtain the json files for the training set and dev set of all datasets.

The default %YOUR_PATH% and %YOUR_OUT_PATH% are both the root path of current repository

1.CNN and DailyMail: extract the question tgz file into a path, then run

`python transform_dataset.py --datatype cnn --path %YOUR_PATH --output_path %YOUR_OUT_PATH%`

It will generate the training and dev json file in SQuAD format.

Optional: In our paper, since these two dataset are too big, we uniformly sample a subset from these two datasets.
The sample interval for CNN training set is 4, while 8 for the both sets of DailyMail. However, we keep the orignal 
CNN dev set.

2.NewsQA: You should put newsqa-data-v1.csv and [cnn_stories.tgz](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ)
 into %YOUR_PATH%, then run
 
`python transform_dataset.py --datatype newsqa --path %YOUR_PATH% --output_path %YOUR_OUT_PATH%`

You will get two files newsqa_train.json and newsqa_dev.json in the output path.

3.CoQA: You need to get the training and dev set of CoQA under a path %YOUR_PATH%, then run

`python transform_dataset.py --datatype coqa --path %YOUR_PATH --output_path %YOUR_OUT_PATH%`

You will get two files coqa_train.json and coqa_dev.json in the output path.

4.DROP: You need to extract the files into a path %YOUR_PATH%, then run

`python transform_dataset.py --datatype drop --path %YOUR_PATH% --output_path %YOUR_OUT_PATH%`

You will get two files drop_train.json and drop_dev.json in the directory.

#### 3. (Optional) Sample subsets from CNN and DailyMail

Since the sizes of these two datasets are obviously larger than other datasets, we sample a subsets 
as the data used in the experiment in the original paper.
We use uniform sampling and the new datasets can be obtained via running

`python sample_dataset.py %FILE_NAME% %OUT_FILE_NAME% %SAMPLE_RATIO%`

In the paper, the sample ratio for DailyMail training and dev set is 8, while 4 for 
CNN training set. CNN dev set remains unchanged due to its small size. 

#### 4. Supervised training on the source domain
To make unsupervised domain adaptation on reading comprehension, you need to run the 
supervised training on the source domain at first. An example for running training

`python run.py
--bert_model bert-base-uncased
--do_train
--do_lower_case
--train_file cnn_train.json
--predict_file cnn_dev.json
--output_dir cnn_models
--output_model_file best_model.bin
--logger_path train_cnn
--use_BN`

'--do_train' indicates running supervised training. You can modified the parameters for --train_file, --predict_file(
the dev file under this mode), --output_dir and --output_model_file under your demand. '--use_BN' means using Batch 
Normalization in the output layer of BERT. We run this on 2 GTX1080Ti GPU with 22GB under the default setting.

#### 5. Unsupervised domain adaptation 
After obtain the supervised training model, you can run adaptation given the source domain
training data, the target domain training data and the target domain dev data. An example is given
for adaptation from CNN to CoQA using entropy weighted CASe (CASe+E).

`python run.py 
--bert_model bert-base-uncased
--do_adaptation
--do_lower_case
--source_train_file cnn_train.json
--target_train_file coqa_train.json
--target_predict_file coqa_dev.json
--input_dir cnn_models
--input_model_file best_model.bin
--output_dir cnn2coqa_models
--output_model_file CASe_model.bin
--logger_path cnn2coqa
--CASe_method CASe+E
--use_BN`

Here, '--do_adaptation' means adaptation mode, you can modify the parameters for '--source_train_file'(training file in 
the source domain), '--target_train_file'(training file in the target domain), '--target_predict_file'(dev file in the 
target domain), '--input_dir'(The path of superivsed trained model on the source domain), '--input_model_file'(name of 
supervised trained model file), 'output_dir'(output path for the adapted model), 'output_model_file'(adapted model file 
name). 

'CASe_method' can be 'CASe' or 'CASe+E', which are standard CASe and CASe with entropy-weighted loss in adversarial
learning respectively.

It should be noted that the process will run evaluation on the target dev set with interval depended on parameter
'--evaluation_interval', which is just used for watching the performance trend. However, the correct result should be the 
performance given by final model which will also be logged.

#### 6. Model prediction

It is used to run prediction for a trained/adapted model. An example is given

`python run.py 
--bert_model bert-base-uncased
--do_prediction
--do_lower_case
--predict_file coqa_dev.json
--output_dir cnn2coqa_models
--output_model_file CASe_model.bin
--logger_path cnn2coqa
--use_BN
--output_prediction`

Here, parameters of '--output_dir' and '--output_model_file' are the model path and model file name for prediction
respectively. '--output_prediction' means a json file with predictions will be written to the output path.

##Still under updating
