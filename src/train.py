import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers.models import llama
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import PreTrainedModel, get_linear_schedule_with_warmup


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
G_INTSEQUENCELENGTH = 2048
G_DICT_NUM_DNA_TABLE = {'Appending': 0, 'N': 1, 'Start': 2, 'End': 3, 'A': 4, 'T': 5, 'C': 6, 'G': 7, 'label_0': 9, 'label_1': 10, '+': 8}

logger = logging.getLogger(__name__)

class LineByLineTextPretrainDatasetJson(Dataset):
    def __init__(self, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, 'llama' + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path) as file:
                self.examples = []
                for line in file:
                    t_listData = json.loads(line.rstrip())
                    if block_size == len(t_listData):
                        self.examples.append(t_listData)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextFinetuneDatasetJson(Dataset):
    def __init__(self, args, file_path, label_path, evaluate=False, block_size=G_INTSEQUENCELENGTH):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, 'llama' + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)

        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path) as file:
                self.examples = []
                for line in file:
                    t_listData = json.loads(line.rstrip())
                    self.examples.append(t_listData)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if label_path is not None:
            with open(label_path) as file:
                self.labels = json.load(file)
        else:
            self.labels = [0 for _ in range(len(self.examples))]
        if len(self.examples) <= len(self.labels):
            self.labels = self.labels[:len(self.examples)]
        else:
            self.examples = self.examples[:len(self.labels)]
        # adjust data size
        if 1.0 < args.ablation_ratio or 0 > args.ablation_ratio:
            raise Exception(f"Error::Current ablation ratio {args.ablation_ratio} is not llegal. Please check it.")
        if not evaluate:
            self.labels = self.labels[:round(len(self.labels) * args.ablation_ratio)]
            self.examples = self.examples[:round(len(self.examples) * args.ablation_ratio)]

        # check amount between sequence and label
        if len(self.examples) != len(self.labels):
            raise Exception(
                f"Sequence({len(self.examples)}) and label({len(self.labels)}) has different quantity, Please check it")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.float)


def load_dataset_from_json(args, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    label_path = args.eval_label_file if evaluate else args.train_label_file
    if 0 == args.model_task:
        return LineByLineTextPretrainDatasetJson(args, file_path=file_path, block_size=args.block_size)
    return LineByLineTextFinetuneDatasetJson(args, file_path, label_path, evaluate, block_size=G_INTSEQUENCELENGTH)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def prepareInputAndLabel(in_tensor: torch.Tensor):
    t_listBatch = deepcopy(in_tensor).tolist()
    inputs = []
    labels = []

    for item in t_listBatch:
        for index in range(len(item) - 1, -1, -1):
            if (len(item) - 1) == index and 0 != item[index]:
                t_listTemp = list(item)
                # t_listTemp[index] = 0
                inputs.append(t_listTemp)
                t_listTemp = item[1:]
                t_listTemp.append(0)
                labels.append(t_listTemp)
                break
            if 0 != item[index] and 0 == item[index + 1]:
                t_listTemp = list(item)
                # t_listTemp[index] = 0
                inputs.append(t_listTemp)
                t_listTemp = item[1:]
                t_listTemp.append(0)
                labels.append(t_listTemp)
                break

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def calculateProcessedToken(in_tensorInput: torch.Tensor):
    return len(torch.nonzero(in_tensorInput))


def reordToken(in_strFile: str, in_floatToken):
    with open(in_strFile, 'a') as out:
        out.write(str(in_floatToken))
        out.write('\n')

def calculatePrediction(in_tensorLogits):
    t_listPrediction = []
    prediction = torch.split(in_tensorLogits, 1, dim=0)
    for item in prediction:
        max_id = torch.argmax(item, dim=2)
        max_id = max_id.tolist()[0]
        t_listPrediction.append(max_id)

    return t_listPrediction

def recordCurrentBatch(in_listPredictions, in_listLabels, in_intBatch, in_strRecordFolder, in_boolEval=False):
    if in_boolEval:
        t_strRecordFile = os.path.join(in_strRecordFolder, 'evalRecording' + str(in_intBatch) + '.txt')
    else:
        t_strRecordFile = os.path.join(in_strRecordFolder, str(in_intBatch) + 'Record.txt')

    if len(in_listLabels) != len(in_listPredictions):
        raise Exception('Predictions and labels has different size,please check it!')
    with open(t_strRecordFile, 'a') as out:
        for index in range(len(in_listLabels)):
            out.write('[Lab]')
            for item in in_listLabels[index]:
                out.write(str(item) + '\t')
            out.write('\n')
            out.write('[Pre]')
            for item in in_listPredictions[index]:
                out.write(str(item) + '\t')
            out.write('\n')

def delAppending(in_listPredictions, in_listLabels):
    t_listPredict = []
    t_listLabel = []
    for index in range(len(in_listLabels)):
        t_intPaddingLength = 0
        for subindex in range(len(in_listLabels[index]) - 1, -1, -1):
            if 0 == in_listLabels[index][subindex]:
                t_intPaddingLength += 1
            else:
                break
        t_listPredict.append(in_listPredictions[index][:len(in_listPredictions[index]) - t_intPaddingLength])
        t_listLabel.append(in_listLabels[index][:len(in_listLabels[index]) - t_intPaddingLength])

    return t_listPredict, t_listLabel

def calculateAccuracyWithoutAppending(in_listPredictions, in_listLabels, in_intBatch, in_strRecordFolder):
    t_strRecordFile = in_strRecordFolder + 'accuracy.txt' if in_strRecordFolder.endswith('/') \
        else in_strRecordFolder + '/accuracy.txt'
    t_floatAccuracy = 0.0
    t_intTokenAmount = 0
    t_listPredictions, t_listLabels = delAppending(in_listPredictions, in_listLabels)
    for index in range(len(t_listLabels)):
        t_intTokenAmount += len(t_listLabels[index])
        if len(t_listPredictions[index]) != len(t_listLabels[index]):
            raise Exception('Prediction and label has different length,please check it!')
        for subindex in range(len(t_listPredictions[index])):
            if 0 == t_listLabels[index][subindex]:
                break
            if t_listPredictions[index][subindex] == t_listLabels[index][subindex]:
                t_floatAccuracy += 1
    t_floatAccuracy = float(t_floatAccuracy) / t_intTokenAmount

    with open(t_strRecordFile, 'a') as out:
        out.write(str(t_floatAccuracy) + '\n')

def getFileNum(in_strFolder: str, in_strPrefix: str = 'evalRecording'):
    t_listDirAndFile = os.listdir(in_strFolder)
    t_listFiles = []
    for file in t_listDirAndFile:
        if os.path.isfile(os.path.join(in_strFolder, file)):
            t_listFiles.append(os.path.splitext(file)[0])  # delete file extension
    num = []
    for file in t_listFiles:
        if file.startswith(in_strPrefix):
            num.append(int(file[13:]))
    if 0 < len(num):  # folder has not recording
        return max(num) + 1  # add 1 to file index
    else:
        return 1  # file index is 1 when folder is vancant

def findCisLabelIndex(in_listInput: list):
    for index in range(len(in_listInput)):
        if index == (len(in_listInput) - 1):
            return index
        elif 0 != in_listInput[index] and 0 == in_listInput[index + 1]:
            return index

def binaryPredictionAndLabel(in_tensorLogits, in_tensorLabels):
    labels = []
    t_intFirstIndex = 0
    firstStack = True
    for lab in in_tensorLabels.tolist():
        labIndex = findCisLabelIndex(lab) - 1
        if 0 == t_intFirstIndex:
            labels.append(lab[labIndex] - 9)
            logits = in_tensorLogits[0][labIndex]
        else:
            labels.append(lab[labIndex] - 9)
            if firstStack:
                logits = torch.stack((logits, in_tensorLogits[t_intFirstIndex][labIndex]), 0)
                firstStack = False
            else:
                logits = torch.cat((logits, in_tensorLogits[t_intFirstIndex][labIndex].view(1, -1)), dim=0)
        t_intFirstIndex += 1

    return logits[:, -3: -1].to("cuda:0").contiguous(), torch.tensor(labels).to("cuda:0").contiguous()

def calculateAcc(in_listPred: list, in_listLab: list):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    if len(in_listLab) != len(in_listPred):
        raise Exception("Error :: can't statistic TP、 FP、 TN、 FN with different size prediction and label")
    for index in range(len(in_listPred)):
        if 1 == in_listPred[index]:
            if 1 == in_listLab[index]:
                tp += 1
            else:
                fp += 1
        else:
            if 0 == in_listLab[index]:
                tn += 1
            else:
                fn += 1
    # calculate acc 、 precision、 recall、 F1
    accuracy = (tp + tn) / len(in_listPred)
    precision = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    f1 = 2 * accuracy * recall / (accuracy + recall + 0.0001)
    indicators = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

    return indicators

def checkEarlyStop(in_listIndicators: list):
    if 5 > len(in_listIndicators):
        return False
    t_listEarlyStop = in_listIndicators[-5:]
    t_floatDiff = 0
    for index in range(4):
        t_floatDiff += t_listEarlyStop[index + 1] - t_listEarlyStop[index]
    if 0.001 >= t_floatDiff:
        return True
    else:
        return False

def train(args, train_dataset, model: PreTrainedModel) -> Tuple[int, float]:
    """ Train the model """
    t_strSummaryWriterFile = os.path.join(args.output_dir, 'SummaryWriter')
    if not os.path.exists(t_strSummaryWriterFile):
        os.makedirs(t_strSummaryWriterFile)
    tb_writer = SummaryWriter(t_strSummaryWriterFile)

    args.train_batch_size = args.per_gpu_train_batch_size * args.n_gpu

    # train_sampler = SequentialSampler(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    set_seed(args)  # Added here for reproducibility
    early_stop = []

    for num in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already mask_tokenstrained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            if 0 == args.model_task:
                inputs, labels = batch, batch
                reordToken(os.path.join(args.output_dir, 'tokenRecord.txt'), calculateProcessedToken(inputs))
            else:
                inputs = batch[0].clone().detach()
                labels = batch[1].clone().detach()

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            logger.info("Step: %d    Loss: %s", step, str(loss))
            tb_writer.add_scalar("train_loss", loss, global_step)

            historyFolder = os.path.join(args.output_dir, 'historyRecord')
            if not os.path.exists(historyFolder):
                os.makedirs(historyFolder)
            if 0 == args.model_task:
                recordCurrentBatch(calculatePrediction(logits), labels.tolist(), num, historyFolder)
                calculateAccuracyWithoutAppending(calculatePrediction(logits), labels.tolist(), num, historyFolder)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if 0 == args.model_task:
                reordToken(os.path.join(args.output_dir, 'lossRecord.txt'), loss)
            loss.squeeze().backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        indicators = evaluate(args, model)
                        for key, value in indicators.items():
                            if 'early_stop' == key:
                                continue
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    if 0 == args.model_task:
                        tb_writer.add_scalar("Perplexity", indicators['perplexity'])
                        reordToken(os.path.join(args.output_dir, 'Perplexity.txt'), indicators['perplexity'])
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            if checkEarlyStop(early_stop) and 0 != args.model_task:
                logger.info("Early stop in step: %d", step)
                epoch_iterator.close()
                break

            if args.max_steps > 0 and global_step > args.max_steps:
                logger.info("Step supass max_steps: %d", step)
                epoch_iterator.close()
                break

        indicators = evaluate(args, model, in_boolEval=True)
        if 0 != args.model_task:
            early_stop.append(indicators['early_stop'])
            if checkEarlyStop(early_stop) and args.model_task not in [-1, 0]:
                train_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            logger.info("Early stop in epoch: %d", num)
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step

def recordR2Epoch(prediction: list, label: list, args, in_boolEval=False):
    if in_boolEval:
        t_strTrainingFile = os.path.join(args.output_dir, 'historyRecord', 'test')
        os.makedirs(t_strTrainingFile, exist_ok=True)
        t_strTrainingFile = os.path.join(t_strTrainingFile,
                                         "R2_Epoch" + str(getFileNum(t_strTrainingFile, 'R2_Epoch')) + '.png')
    else:
        t_strTrainingFile = os.path.join(args.output_dir, 'historyRecord', 'eval')
        os.makedirs(t_strTrainingFile, exist_ok=True)
        t_strTrainingFile = os.path.join(t_strTrainingFile,
                                         "R2_Eval" + str(getFileNum(t_strTrainingFile, 'R2_Eval')) + '.png')

    r2 = r2_score(label, prediction)
    plt.scatter(prediction, label, color='blue', label=f"R² ={r2:.3f}")
    # plt.plot(predict, label, color='red', label=f"Linear Regression (R² ={r2:.2f})")

    plt.legend()

    # plt.title('Linear Regression with R²')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig(t_strTrainingFile)
    plt.close()

    return r2

def filterPredAndLab(predictions: list, labels: list, minShreshold=-0.5, maxShreshold=0.5):
    filterPreds = []
    filterLabels = []
    for index in range(len(predictions)):
        if (predictions[index] > minShreshold and predictions[index] < maxShreshold) or (
                labels[index] > minShreshold and labels[index] < maxShreshold):
            continue
        filterPreds.append(predictions[index])
        filterLabels.append(labels[index])

    return filterPreds, filterLabels

def calIndAccuracyR2(predictions: list, labels: list):
    if len(predictions) != len(labels):
        print(f"Error:: aclAccuracy: the input predictions and labels has different size: "
              f"predictions({len(predictions)}) and labels({len(labels)})")
    positive = 0
    negative = 0
    for index in range(len(predictions)):
        if predictions[index] > 0 and labels[index] > 0:
            positive += 1
        elif predictions[index] < 0 and labels[index] < 0:
            negative += 1

    if 0 == len(predictions):
        return 0

    return (positive + negative)/len(predictions)

def getDecoderTable(encoderTab: dict):
    decoderTab = {}
    for key, value in encoderTab.items():
        decoderTab[value] = key
    return decoderTab

def decoderSeq(numSeq: list, numTab: dict):
    decoderTab = getDecoderTable(numTab)
    decoderSequence = []
    for seq in numSeq:
        deSeq = ''
        for item in seq:
            if item in decoderTab:
                if 0 == item:
                    continue
                deSeq += decoderTab[item]
            else:
                print(f"Error:: loss number base {item}")
        decoderSequence.append(deSeq)

    return decoderSequence

def evaluate(args, model: PreTrainedModel, in_intEpoch = 0, in_boolEval=False, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_dataset_from_json(args, evaluate=True)

    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * args.n_gpu

    # Note that DistributedSampler samples randomly
    if in_boolEval:
        eval_samplerTrain = SequentialSampler(eval_dataset)
    else:
        eval_samplerTrain = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_samplerTrain, batch_size=args.eval_batch_size
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    allIndicators = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    allPrediction = []
    allLabel = []
    indicators = {}
    t_strOutputFolder = os.path.join(args.output_dir, 'historyRecord')
    recordingNum = getFileNum(t_strOutputFolder)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        if 0 == args.model_task:
            inputs, labels = batch, batch
        else:
            inputs = batch[0].clone().detach()
            labels = batch[1].clone().detach()
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss, logits = outputs[:2]
            if 0 == args.model_task:
                reordToken(os.path.join(args.output_dir, 'testLossRecord.txt'), lm_loss)
                if in_boolEval:
                    recordCurrentBatch(calculatePrediction(logits), labels.tolist(),
                                       recordingNum, t_strOutputFolder, in_boolEval=True)
            eval_loss += lm_loss.mean().item()
            # recording current prediction

            if 0 != args.model_task:
                for pred in logits.tolist():
                    for subPred in pred:
                        allPrediction.append(subPred)
                allLabel.extend(labels.tolist())
                # record label, prediction and loss in this batch by log
                # logger.info(f"Eval:: Label: {labels}")
                logger.info(f"Eval:: input: {inputs}")
                logger.info(f'Eval:: Prediction: {logits}')
                logger.info(f'Eval:: Loss: {lm_loss}')

                if in_boolEval:
                    with open(os.path.join(args.output_dir, 'historyRecord', 'trainingRecord.txt'), 'a+') as out:
                        out.write('Prediction:\t')
                        temp = '\t'.join(map(str, logits.tolist()))
                        out.write(temp + '\n')
                        out.write('Label:\t')
                        temp = '\t'.join(map(str, labels.tolist()))
                        out.write(temp + '\n')
                    with open(os.path.join(args.output_dir, 'historyRecord', 'evalLossRecord.txt'), 'a+') as out_loss:
                        out_loss.write("Test\t" + str(lm_loss) + '\n')
                else:
                    with open(os.path.join(args.output_dir, 'historyRecord', 'evalLossRecord.txt'), 'a+') as out:
                        out.write("Eval\t" + str(lm_loss) + '\n')
        nb_eval_steps += 1
        if not in_boolEval:
            break

    if 0 != args.model_task and in_boolEval:
        r2 = recordR2Epoch(allPrediction, allLabel, args)
        r2 = float(r2)
        # 计算定性指标
        allPrediction, allLabel = filterPredAndLab(allPrediction, allLabel)
        acc = calIndAccuracyR2(allPrediction, allLabel)
        logger.info(f"Eval:: Promoter strength is  {acc}")
        # save Epoch model
        checkpoint_prefix = "epochCheckpoint"
        # Save model checkpoint
        r2_file = os.path.join(args.output_dir, 'r2Record.json')
        if os.path.exists(r2_file):
            with open(r2_file) as file:
                r2_record = json.load(file)
                r2_record.append(r2)
        else:
            r2_record = [r2]
        with open(r2_file, 'w') as out:
            out.write(json.dumps(r2_record) + '\n')
        if 1 == len(r2_record) or r2 > max(r2_record[:-1]):
            output_dir = os.path.join(args.output_dir, 'epochModel')
            os.makedirs(output_dir, exist_ok=True)
            output_dir = os.path.join(output_dir, "{}-{}".format(checkpoint_prefix, in_intEpoch))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            # tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
        indicators = {'R2': r2}

        indicators['early_stop'] = indicators['R2']
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        indicators["perplexity"] = perplexity
    else:
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        indicators["perplexity"] = perplexity

    return indicators

def eval(args, model: PreTrainedModel, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if 0 == args.model_task:
        raise Exception(f"Only during the fine-tuning phase can predictions be made directly.")
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_dataset = load_dataset_from_json(args, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * args.n_gpu

    # prepare for dataset
    eval_samplerTrain = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_samplerTrain, batch_size=args.eval_batch_size
    )
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    model.eval()
    allPrediction = []
    allLabel = []
    # output_num = getFileNum(os.path.join(args.output_dir, 'historyRecord'), 'trainingRecord_')
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch[0].clone().detach().to(args.device)
        labels = batch[1].clone().detach().to(args.device)

        with torch.no_grad(): 
            outputs = model(inputs, labels=labels)
            lm_loss, logits = outputs[:2]
            eval_loss += lm_loss.mean().item()
            for pred in logits.tolist():
                for subPred in pred:
                    allPrediction.append(subPred)
            allLabel.extend(labels.tolist())
            # record label, prediction and loss in this batch by log
            deSequences = decoderSeq(inputs.tolist(), G_DICT_NUM_DNA_TABLE)
            logger.info(f"Eval:: input: {deSequences[0]}")
            logger.info(f'Eval:: Prediction: {logits}')
            logger.info(f'Eval:: Loss: {lm_loss}')
            outFileName = Path(args.eval_data_file).stem + '_trainingRecord.txt'
            with open(os.path.join(args.output_dir, outFileName), 'a+') as out:
                pred = logits.tolist()
                if len(deSequences) != len(pred):
                    logger.info("Error:: can't record sequence and prediction with different size")
                for index in range(len(deSequences)):
                    out.write(deSequences[index] + '\t' + str(pred[index][0]) + '\n')

