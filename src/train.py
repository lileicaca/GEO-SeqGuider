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

class lineByLineTextPretrainDatasetJson(Dataset):
    def __init__(self, args, filePath: str, blockSize=G_INTSEQUENCELENGTH):
        assert os.path.isfile(filePath)

        logger.info("Load pretraining dataset from file %s", filePath)
        with open(filePath) as file:
            self.examples = []
            for line in file:
                tempData = json.loads(line.rstrip())
                if blockSize == len(tempData):
                    self.examples.append(tempData)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class lineByLineTextFinetuneDatasetJson(Dataset):
    def __init__(self, args, filePath, labelPath, evaluate=False, blockSize=G_INTSEQUENCELENGTH):
        assert os.path.isfile(filePath)

        logger.info("Load fine-tuning dataset from file %s", filePath)
        with open(filePath) as file:
            self.examples = []
            for line in file:
                tempData = json.loads(line.rstrip())
                self.examples.append(tempData)

        if labelPath is not None:
            with open(labelPath) as file:
                self.labels = json.load(file)
        else:
            self.labels = [0 for _ in range(len(self.examples))]
        if len(self.examples) != len(self.labels):
            raise Exception(f"The number of sequences and labels is not equal, please check it.")
        # adjust data size
        if not 0 < args.ablation_ratio <= 1.0:
            raise Exception(f"Error::Current ablation ratio {args.ablation_ratio} is not llegal. Please check it.")
        if not evaluate:
            self.labels = self.labels[:round(len(self.labels) * args.ablation_ratio)]
            self.examples = self.examples[:round(len(self.examples) * args.ablation_ratio)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.float)


def loadDatasetFromJson(args, evaluate=False):
    filePath = args.eval_data_file if evaluate else args.train_data_file
    labelPath = args.eval_label_file if evaluate else args.train_label_file
    if 0 == args.model_task:
        return lineByLineTextPretrainDatasetJson(args, filePath=filePath, blockSize=args.block_size)
    return lineByLineTextFinetuneDatasetJson(args, filePath, labelPath, evaluate, blockSize=G_INTSEQUENCELENGTH)

def randomSeed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def sortedCheckpoints(args, checkpointPrefix="checkpoint") -> List[str]:
    checkpoints = []

    globCheckpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpointPrefix)))

    for path in globCheckpoints:
        regMatch = re.match(".*{}-([0-9]+)".format(checkpointPrefix), path)
        if regMatch and regMatch.groups():
            checkpoints.append((int(regMatch.groups()[0]), path))

    sortedCheckpoints = sorted(checkpoints)
    sortedCheckpoints = [checkpoint[1] for checkpoint in sortedCheckpoints]
    return sortedCheckpoints

def rotateCheckpoints(args, checkpointPrefix="checkpoint") -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    sortedCheckpoints = sortedCheckpoints(args, checkpointPrefix)
    if len(sortedCheckpoints) <= args.save_total_limit:
        return

    deleteCount = max(0, len(sortedCheckpoints) - args.save_total_limit)
    deleteCheckpoints = sortedCheckpoints[:deleteCount]
    for checkpoint in deleteCheckpoints:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def calculateProcessedToken(tensorInput: torch.Tensor):
    return len(torch.nonzero(tensorInput))


def reordToFile(file: str, token):
    with open(file, 'a') as out:
        out.write(str(token))
        out.write('\n')

def calculatePrediction(tensorLogits):
    tempPredict = []
    prediction = torch.split(tensorLogits, 1, dim=0)
    for item in prediction:
        maxId = torch.argmax(item, dim=2)
        maxId = maxId.tolist()[0]
        tempPredict.append(maxId)

    return tempPredict

def recordCurrentBatch(predictions, labels, batch, recordDir, boolEval=False):
    if boolEval:
        recordFile = os.path.join(recordDir, 'evalRecording' + str(batch) + '.txt')
    else:
        recordFile = os.path.join(recordDir, str(batch) + 'Record.txt')

    if len(labels) != len(predictions):
        raise Exception('Predictions and labels has different size,please check it!')
    with open(recordFile, 'a') as out:
        for index in range(len(labels)):
            out.write('[Lab]')
            for item in labels[index]:
                out.write(str(item) + '\t')
            out.write('\n')
            out.write('[Pre]')
            for item in predictions[index]:
                out.write(str(item) + '\t')
            out.write('\n')

def delAppending(predictions, labels):
    tempPredict = []
    tempLabel = []
    for index in range(len(labels)):
        paddingLength = 0
        for subindex in range(len(labels[index]) - 1, -1, -1):
            if 0 == labels[index][subindex]:
                paddingLength += 1
            else:
                break
        tempPredict.append(predictions[index][:len(predictions[index]) - paddingLength])
        tempLabel.append(labels[index][:len(labels[index]) - paddingLength])

    return tempPredict, tempLabel

def calculateAccuracyWithoutAppending(predictions, labels, batch, recordDir):
    recordFile = recordDir + 'accuracy.txt' if recordDir.endswith('/') \
        else recordDir + '/accuracy.txt'
    acc = 0.0
    tokenAmount = 0
    predictions, labels = delAppending(predictions, labels)
    for index in range(len(labels)):
        tokenAmount += len(labels[index])
        if len(predictions[index]) != len(labels[index]):
            raise Exception('Prediction and label has different length,please check it!')
        for subindex in range(len(predictions[index])):
            if 0 == labels[index][subindex]:
                break
            if predictions[index][subindex] == labels[index][subindex]:
                acc += 1
    acc = float(acc) / tokenAmount

    with open(recordFile, 'a') as out:
        out.write(str(acc) + '\n')

def getFileNum(recordFolder: str, filePrefix: str = 'evalRecording'):
    dirAndFile = os.listdir(recordFolder)
    files = []
    for file in dirAndFile:
        if os.path.isfile(os.path.join(recordFolder, file)):
            files.append(os.path.splitext(file)[0])  # delete file extension
    num = []
    for file in files:
        if file.startswith(filePrefix):
            num.append(int(file[13:]))
    if 0 < len(num):  # folder has not recording
        return max(num) + 1  # add 1 to file index
    else:
        return 1  # file index is 1 when folder is vancant

def calculateAcc(pred: list, label: list):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    if len(label) != len(pred):
        raise Exception("Error :: can't statistic TP、 FP、 TN、 FN with different size prediction and label")
    for index in range(len(pred)):
        if 1 == pred[index]:
            if 1 == label[index]:
                tp += 1
            else:
                fp += 1
        else:
            if 0 == label[index]:
                tn += 1
            else:
                fn += 1
    # calculate acc 、 precision、 recall、 F1
    accuracy = (tp + tn) / len(pred)
    precision = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    f1 = 2 * accuracy * recall / (accuracy + recall + 0.0001)
    indicators = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

    return indicators

def checkEarlyStop(indicators: list):
    if 5 > len(indicators):
        return False
    earlyStop = indicators[-5:]
    diff = 0
    for index in range(4):
        diff += earlyStop[index + 1] - earlyStop[index]
    if 0.001 >= diff:
        return True
    else:
        return False

def recordR2Epoch(prediction: list, label: list, args, boolEval=False):
    if boolEval:
        outputFile = os.path.join(args.output_dir, 'historyRecord', 'test')
        os.makedirs(outputFile, exist_ok=True)
        outputFile = os.path.join(outputFile,
                                         "R2_Epoch" + str(getFileNum(outputFile, 'R2_Epoch')) + '.png')
    else:
        outputFile = os.path.join(args.output_dir, 'historyRecord', 'eval')
        os.makedirs(outputFile, exist_ok=True)
        outputFile = os.path.join(outputFile,
                                         "R2_Eval" + str(getFileNum(outputFile, 'R2_Eval')) + '.png')

    r2 = r2_score(label, prediction)
    plt.scatter(prediction, label, color='blue', label=f"R² ={r2:.3f}")
    # plt.plot(predict, label, color='red', label=f"Linear Regression (R² ={r2:.2f})")

    plt.legend()

    # plt.title('Linear Regression with R²')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig(outputFile)
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

def train(args, trainDataset, model: PreTrainedModel) -> Tuple[int, float]:
    """ Training the model """
    summaryWriterFile = os.path.join(args.output_dir, 'SummaryWriter')
    if not os.path.exists(summaryWriterFile):
        os.makedirs(summaryWriterFile)
    tbWriter = SummaryWriter(summaryWriterFile)

    trainSampler = RandomSampler(trainDataset)
    trainDataloader = DataLoader(
        trainDataset, sampler=trainSampler, batch_size=args.train_batch_size
    )
    totalStep = len(trainDataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    noDecay = ["bias", "LayerNorm.weight"]
    optimizerParameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in noDecay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in noDecay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizerParameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=totalStep
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(trainDataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", totalStep)

    globalStep = 0
    trLoss, loggingLoss = 0.0, 0.0
    model.zero_grad()
    epochIterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    randomSeed(args)  # Added here for reproducibility
    earlyStop = []

    for num in epochIterator:
        batchIterator = tqdm(trainDataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(batchIterator):
            if 0 == args.model_task:
                inputs, labels = batch, batch
                reordToFile(os.path.join(args.output_dir, 'tokenRecord.txt'), calculateProcessedToken(inputs))
            else:
                inputs, labels = (batch[0].clone().detach(), batch[1].clone().detach())

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss, logits = outputs[:2]
            logger.info("Step: %d    Loss: %s", step, str(loss))
            tbWriter.add_scalar("train_loss", loss, globalStep)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if 0 == args.model_task:
                reordToFile(os.path.join(args.output_dir, 'lossRecord.txt'), loss)
            loss.squeeze().backward()

            trLoss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                globalStep += 1

                if args.logging_steps > 0 and globalStep % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        # Use a batch of data from the test set to test the performance of the current model
                        indicators = evaluate(args, model)
                        for key, value in indicators.items():
                            if 'early_stop' == key:
                                continue
                            tbWriter.add_scalar("eval_{}".format(key), value, globalStep)
                    tbWriter.add_scalar("lr", scheduler.get_lr()[0], globalStep)
                    tbWriter.add_scalar("loss", (trLoss - loggingLoss) / args.logging_steps, globalStep)
                    if 0 == args.model_task:
                        tbWriter.add_scalar("Perplexity", indicators['perplexity'])
                        reordToFile(os.path.join(args.output_dir, 'Perplexity.txt'), indicators['perplexity'])
                    loggingLoss = trLoss

                if args.save_steps > 0 and globalStep % args.save_steps == 0:
                    checkpointPrefix = "checkpoint"
                    # Save model checkpoint
                    outputDir = os.path.join(args.output_dir, "{}-{}".format(checkpointPrefix, globalStep))
                    os.makedirs(outputDir, exist_ok=True)
                    model.save_pretrained(outputDir)

                    torch.save(args, os.path.join(outputDir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", outputDir)

                    rotateCheckpoints(args, checkpointPrefix)

                    torch.save(optimizer.state_dict(), os.path.join(outputDir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(outputDir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", outputDir)
            if checkEarlyStop(earlyStop) and 0 != args.model_task:
                logger.info("Early stop in step: %d", step)
                batchIterator.close()
                break

        # Use all the data in the test set to test the performance of the current model
        indicators = evaluate(args, model, boolEval=True)
        if 0 != args.model_task:
            earlyStop.append(indicators['early_stop'])
            if checkEarlyStop(earlyStop) and args.model_task not in [-1, 0]:
                epochIterator.close()
                break

    tbWriter.close()

    return globalStep, trLoss / globalStep

def evaluate(args, model: PreTrainedModel, epochNum = 0, boolEval=False, prefix="") -> Dict:
    evalDataset = loadDatasetFromJson(args, evaluate=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # If testing model performance, make predictions in order; otherwise, shuffle the order of data
    if boolEval:
        evalSampler = SequentialSampler(evalDataset)
    else:
        evalSampler = RandomSampler(evalDataset)
    evalDataLoader = DataLoader(
        evalDataset, sampler=evalSampler, batch_size=args.eval_batch_size
    )

    # Eval!
    logger.info("Running evaluation {}".format(prefix))
    logger.info("  Num examples = %d", len(evalDataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    evalLoss = 0.0
    evalStep = 0
    model.eval()
    allIndicators = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    allPrediction = []
    allLabel = []
    indicators = {}
    outputDir = os.path.join(args.output_dir, 'historyRecord')
    recordingNum = getFileNum(outputDir)
    for batch in tqdm(evalDataLoader, desc="Evaluating"):
        if 0 == args.model_task:
            inputs, labels = batch, batch
        else:
            inputs, labels = (batch[0].clone().detach(), batch[1].clone().detach())
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lmLoss, logits = outputs[:2]
            if 0 == args.model_task:
                reordToFile(os.path.join(args.output_dir, 'testLossRecord.txt'), lmLoss)
            evalLoss += lmLoss.mean().item()
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
                logger.info(f'Eval:: Loss: {lmLoss}')

                if boolEval:
                    with open(os.path.join(args.output_dir, 'historyRecord', 'trainingRecord.txt'), 'a+') as out:
                        out.write('Prediction:\t')
                        temp = '\t'.join(map(str, logits.tolist()))
                        out.write(temp + '\n')
                        out.write('Label:\t')
                        temp = '\t'.join(map(str, labels.tolist()))
                        out.write(temp + '\n')
                    with open(os.path.join(args.output_dir, 'historyRecord', 'evalLossRecord.txt'), 'a+') as out_loss:
                        out_loss.write("Test\t" + str(lmLoss) + '\n')
                else:
                    with open(os.path.join(args.output_dir, 'historyRecord', 'evalLossRecord.txt'), 'a+') as out:
                        out.write("Eval\t" + str(lmLoss) + '\n')
        evalStep += 1
        # During the model training process, only one batch of data from the test set is used as the validation set to test the performance of the model
        if not boolEval:
            break
    # If fine-tuning is made, use r2 to evaluate the performance of the model
    if 0 != args.model_task and boolEval:
        r2 = recordR2Epoch(allPrediction, allLabel, args)
        r2 = float(r2)
        # Calculate qualitative indicators
        allPrediction, allLabel = filterPredAndLab(allPrediction, allLabel)
        acc = calIndAccuracyR2(allPrediction, allLabel)
        logger.info(f"Eval:: The accuracy of predicting the upregulation and downregulation of bacterial expression levels is:  {acc}")
        # save Epoch model
        checkpointPrefix = "epochCheckpoint"
        # Save model checkpoint
        r2File = os.path.join(args.output_dir, 'r2Record.json')
        if os.path.exists(r2File):
            with open(r2File) as file:
                r2Record = json.load(file)
                r2Record.append(r2)
        else:
            r2Record = [r2]
        with open(r2File, 'w') as out:
            out.write(json.dumps(r2Record) + '\n')
        if 1 == len(r2Record) or r2 > max(r2Record[:-1]):
            outputDir = os.path.join(args.output_dir, 'epochModel')
            os.makedirs(outputDir, exist_ok=True)
            outputDir = os.path.join(outputDir, "{}-{}".format(checkpointPrefix, epochNum))
            os.makedirs(outputDir, exist_ok=True)
            model.save_pretrained(outputDir)
            torch.save(args, os.path.join(outputDir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", outputDir)
        indicators = {'R2': r2}

        indicators['early_stop'] = indicators['R2']
        evalLoss = evalLoss / evalStep
        perplexity = torch.exp(torch.tensor(evalLoss))
        indicators["perplexity"] = perplexity
    else:
        evalLoss = evalLoss / evalStep
        perplexity = torch.exp(torch.tensor(evalLoss))
        indicators["perplexity"] = perplexity

    return indicators

def eval(args, model: PreTrainedModel, prefix="") -> Dict:
    if 0 == args.model_task:
        raise Exception(f"Only during the fine-tuning phase can predictions be made directly.")
    os.makedirs(args.output_dir, exist_ok=True)
    evalDataset = loadDatasetFromJson(args, evaluate=True)

    # prepare for dataset
    evalSampler = SequentialSampler(evalDataset)
    evalDataLoader = DataLoader(
        evalDataset, sampler=evalSampler, batch_size=args.eval_batch_size
    )
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(evalDataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    evalLoss = 0.0
    model.eval()
    allPrediction = []
    allLabel = []
    for batch in tqdm(evalDataLoader, desc="Evaluating"):
        inputs = batch[0].clone().detach().to(args.device)
        labels = batch[1].clone().detach().to(args.device)

        with torch.no_grad(): 
            outputs = model(inputs, labels=labels)
            lmLoss, logits = outputs[:2]
            evalLoss += lmLoss.mean().item()
            for pred in logits.tolist():
                for subPred in pred:
                    allPrediction.append(subPred)
            allLabel.extend(labels.tolist())
            # record label, prediction and loss in this batch by log
            deSequences = decoderSeq(inputs.tolist(), G_DICT_NUM_DNA_TABLE)
            logger.info(f"Eval:: input: {deSequences[0]}")
            logger.info(f'Eval:: Prediction: {logits}')
            logger.info(f'Eval:: Loss: {lmLoss}')
            outFileName = Path(args.eval_data_file).stem + '_trainingRecord.txt'
            with open(os.path.join(args.output_dir, outFileName), 'a+') as out:
                pred = logits.tolist()
                if len(deSequences) != len(pred):
                    logger.info("Error:: can't record sequence and prediction with different size")
                for index in range(len(deSequences)):
                    out.write(deSequences[index] + '\t' + str(pred[index][0]) + '\n')

