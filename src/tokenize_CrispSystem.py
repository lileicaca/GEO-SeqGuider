import os
from pathlib import Path
import datetime
import json
import random
import pandas as pd
import argparse

# tokenizer vocabulary
G_DICT_NUM_DNA_TABLE = {'Appending': 0, 'N': 1, 'Start': 2, 'End': 3, 'A': 4, 'T': 5, 'C': 6,
                        'G': 7, 'Special': 8, 'label_0': 9, 'label_1': 10, '+': 8}


def checkSeq(seq: str, legalBase=['A', 'T', 'C', 'G']):
    for base in seq.upper():
        if base not in legalBase:
            print(f"[checkSeq] Warnning:: illegal base {base}")
            return False
    return True


def numSequence(in_listSequence: list, in_dictNumTable: dict = G_DICT_NUM_DNA_TABLE,
                in_intLength=2048):
    numberDNA = []
    for dna in in_listSequence:
        currentSeq = []
        for alphabet in dna:
            if alphabet not in in_dictNumTable:
                print(f"Error :: can't find current vocabulary {alphabet}")
                break
            else:
                currentSeq.append(in_dictNumTable[alphabet])
        if len(dna) == len(currentSeq):
            while len(currentSeq) < in_intLength:
                currentSeq.append(in_dictNumTable['Appending'])
            numberDNA.append(currentSeq)
        else:
            print(f"Loss sequence {dna}")
    return numberDNA


def shuffleList(seq: list, label: list):
    combinedDataset = list(zip(seq, label))
    random.shuffle(combinedDataset)
    seq, label = zip(*combinedDataset)

    return list(seq), list(label)


def readSGRN(in_strFile: Path, columns=['LogZ_Log_delta_1', 'LogZ_Log_delta_2', 'LogZ_Log_delta_3', 'LogZ_Log_delta_4']):
    data = pd.read_excel(in_strFile, sheet_name='crisp')
    resultTrain = {'sequence': [], 'label': []}
    resultTest = {'sequence': [], 'label': []}
    promoter = data['Promoter']
    sgRNA = data['sgRNA']
    pam = data['PAM']
    label1 = data[columns[0]]
    label2 = data[columns[1]]
    label3 = data[columns[2]]
    label4 = data[columns[3]]

    promoterSeq = ''
    testFlag = False
    for index in range(len(promoter)):
        if str(promoter[index]).strip().startswith('LongPromoter') or 'None' == str(promoter[index]).strip():    # First line
            continue
        if '无' == sgRNA[index]:
            promoterSeq = promoter[index]
            testFlag = True
            continue
        else:
            if not checkSeq(promoterSeq):
                print(f"Illegal Promoter: {promoterSeq}")
            if isinstance(sgRNA[index], (int, float)) or not checkSeq(str(sgRNA[index].strip())):
                print(f"Illegal sgRNA: {sgRNA[index]}")
            if isinstance(pam[index], (int, float)) or not checkSeq(str(pam[index].strip())):
                print(f"Illegal PAM: {pam[index]}")
            temSeq = promoterSeq.strip().upper() + '+' + sgRNA[index].strip().upper() + '+' + pam[index].strip().upper()    # with special token
            print(f"Label : {label1[index], label2[index], label3[index], label4[index]}")
            if not pd.isna(label1[index]) and not pd.isna(label2[index]) and\
                not pd.isna(label3[index]) and not pd.isna(label4[index]):
                if testFlag:
                    resultTest['sequence'].append(temSeq.upper())
                    lab = [label1[index], label2[index], label3[index], label4[index]]
                    random.shuffle(lab)
                    resultTest['label'].append(lab[0])
                    testFlag = False
                else:
                    resultTrain['sequence'].append(temSeq.upper())
                    resultTrain['label'].append([label1[index], label2[index], label3[index], label4[index]])

    return resultTrain, resultTest

def readSGRNForPrediction(in_strFile: Path):
    data = pd.read_excel(in_strFile, sheet_name='crisp')
    resultDataSet = []
    promoter = data['Promoter']
    sgRNA = data['sgRNA']
    pam = data['PAM']
    promoterSeq = ''
    testFlag = False
    for index in range(len(promoter)):
        if str(promoter[index]).strip().startswith('LongPromoter') or 'None' == str(promoter[index]).strip():    # First line
            continue
        if '无' == sgRNA[index]:
            promoterSeq = promoter[index]
            testFlag = True
            continue
        else:
            if not checkSeq(promoterSeq):
                print(f"Illegal Promoter: {promoterSeq}")
            if isinstance(sgRNA[index], (int, float)) or not checkSeq(str(sgRNA[index].strip())):
                print(f"Illegal sgRNA: {sgRNA[index]}")
            if isinstance(pam[index], (int, float)) or not checkSeq(str(pam[index].strip())):
                print(f"Illegal PAM: {pam[index]}")
            temSeq = promoterSeq.strip().upper() + '+' + sgRNA[index].strip().upper() + '+' + pam[index].strip().upper()    # with special token
            if testFlag:
                resultDataSet.append(temSeq.upper())


    return resultDataSet


def numLB(dataFile: Path, outFolder: Path):
    tpmColumns = ['Lablel-1', 'Lablel-2', 'Lablel-3', 'Lablel-4']
    resTrain, resTest = readSGRN(dataFile, tpmColumns)
    print(f'Training set size is {len(resTrain)}, Test set size is {len(resTest)}')
    t_strOutputFolder = outFolder
    # training dataset
    numsequence = numSequence(resTrain["sequence"], G_DICT_NUM_DNA_TABLE, 2048)
    if len(numsequence) != len(resTrain['label']):
        raise Exception("Loss some sequence after numbering dna")
    # shuffle training dataset
    combinedDataset = list(zip(numsequence, resTrain['label']))
    random.shuffle(combinedDataset)
    numsequence, resTrain['label'] = zip(*combinedDataset)
    numsequence = list(numsequence)
    resTrain['label'] = list(resTrain['label'])
    # extract training dataset
    trainingSet = []
    trainingLab = []
    for index in range(len(numsequence)):
        for lab in resTrain['label'][index]:
            trainingSet.append(numsequence[index])
            trainingLab.append(lab)
    trainingSet, trainingLab = shuffleList(trainingSet, trainingLab)
    # test dataset
    numsequence = numSequence(resTest["sequence"], G_DICT_NUM_DNA_TABLE, 2048)
    if len(numsequence) != len(resTest['label']):
        raise Exception("Loss some sequence after numbering dna")
    testSet = []
    testLab = []
    for index in range(len(numsequence)):
        testSet.append(numsequence[index])
        testLab.append(resTest["label"][index])
    testSet, testLab = shuffleList(testSet, testLab)
    t_strOutputFile = os.path.join(t_strOutputFolder, 'training_gene.jsonl')
    with open(t_strOutputFile, 'w') as out:
        for seq in trainingSet:
            out.write(json.dumps(seq) + '\n')
    t_strOutputFile = os.path.join(t_strOutputFolder, 'trainingLab_gene.json')
    with open(t_strOutputFile, 'w') as out:
        out.write(json.dumps(trainingLab) + '\n')
    t_strOutputFile = os.path.join(t_strOutputFolder, 'test_gene.jsonl')
    with open(t_strOutputFile, 'w') as out:
        for seq in testSet:
            out.write(json.dumps(seq) + '\n')
    t_strOutputFile = os.path.join(t_strOutputFolder, 'testLab_gene.json')
    with open(t_strOutputFile, 'w') as out:
        out.write(json.dumps(testLab) + '\n')

def numForPrediction(dataFile: Path, outFolder: Path):
    sequences = readSGRNForPrediction(dataFile)
    print(f'Data set size is {len(sequences)}')
    t_strOutputFolder = outFolder
    # numerate dataset
    numsequence = numSequence(sequences, G_DICT_NUM_DNA_TABLE, 2048)
    t_strOutputFile = os.path.join(t_strOutputFolder, 'predict_gene.jsonl')
    with open(t_strOutputFile, 'w') as out:
        for seq in numsequence:
            out.write(json.dumps(seq) + '\n')
