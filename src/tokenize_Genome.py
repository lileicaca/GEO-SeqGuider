import os
import random
import json
import datetime
import argparse

# Principle of complementary base pairing
G_DIC_BASETAB = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
# tokenizer vocabulary
G_DICT_NUM_DNA_TABLE = {'Appending': 0, 'N': 1, 'Start': 2, 'End': 3, 'A': 4, 'T': 5, 'C': 6,
                        'G': 7, 'Special': 8, 'label_0': 9, 'label_1': 10}
G_INT_SEQUENCELENGTH = 2048


def getCompleChain(dna: str):
    comDNA = ''
    for index in range(len(dna)):
        if dna[index] in G_DIC_BASETAB:
            comDNA += G_DIC_BASETAB[dna[index]]
        else:
            print(f"Warning:: unknow base {dna[index]}")
            comDNA += 'N'
    return comDNA[::-1]


def numSequence(in_listSequence: list, in_dictNumTable: dict):
    numberDNA = []
    for dna in in_listSequence:
        currentSeq = []
        for alphabet in dna:
            if alphabet not in in_dictNumTable:
                print(f"Warning :: can't find current vocabulary {alphabet}")
                currentSeq.append(in_dictNumTable['N'])  # Nucleotide 'N' replace all illegal base
            else:
                currentSeq.append(in_dictNumTable[alphabet])
        if len(dna) == len(currentSeq):
            currentSeq.insert(0, in_dictNumTable['Start'])
            currentSeq.append(in_dictNumTable['End'])
            numberDNA.append(currentSeq)
        else:
            print(f"Loss sequence {dna[:100]}")
    return numberDNA


def readAllSequence(in_strCdsFile: str):
    seq = {}
    with open(in_strCdsFile) as file:
        seqName = ''
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if 0 < len(seqName):
                    seq[seqName] = sequence.upper()
                    seq[seqName + 'reverse'] = getCompleChain(sequence.upper())
                    sequence = ''
                    seqName = line.strip()[1:].split()[0]
                else:
                    sequence += line.strip()
            if seqName not in seq:
                seq[seqName] = sequence.upper()
                seq[seqName + 'reverse'] = getCompleChain(sequence.upper())
    return seq


def splitDNAsequence(in_listNumDNA: list, in_dictNumTable: dict, in_intLength=512):
    newDNASequence = []
    for seq in in_listNumDNA:
        for index in range(len(seq) // in_intLength):
            newDNASequence.append(seq[in_intLength * index: in_intLength * (index + 1)])
        if 0 == (len(seq) % in_intLength):
            continue
        lastSeq = seq[(-(len(seq) % in_intLength)):]
        while in_intLength > len(lastSeq):
            lastSeq.append(in_dictNumTable['Appending'])
        newDNASequence.append(lastSeq)
    return newDNASequence


def numGenome(genome='data/kt2440.fa', outputDir='input'):
    DNASequence = readAllSequence(genome)
    numberDNA = numSequence(list(DNASequence.values()), G_DICT_NUM_DNA_TABLE)
    numberDNA = splitDNAsequence(numberDNA, G_DICT_NUM_DNA_TABLE, G_INT_SEQUENCELENGTH)
    random.shuffle(numberDNA)
    with open(os.path.join(outputDir, f'train_numbase_{G_INT_SEQUENCELENGTH}.jsonl'),
              'w') as out:
        for seq in numberDNA[: -25]:
            out.write(json.dumps(seq) + '\n')
    with open(os.path.join(outputDir, f'test_numbase_{G_INT_SEQUENCELENGTH}.jsonl'),
              'w') as out:
        for seq in numberDNA[-25:]:
            out.write(json.dumps(seq) + '\n')
