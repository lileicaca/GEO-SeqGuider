import argparse
import logging
import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers.models import llama
import matplotlib
matplotlib.use("Agg")
from transformers import PreTrainedModel

G_INT_SEQUENCELENGTH = 2048
G_DICT_NUM_DNA_TABLE = {'Appending': 0, 'N': 1, 'Start': 2, 'End': 3, 'A': 4, 'T': 5, 'C': 6, 'G': 7, 'label_0': 9, 'label_1': 10, '+': 8}
G_DEFAULT_PROMOTER = 'CCCATGCTGGCGTTGGTGCTGACCCTGTGTGGCGGTGCGATGTGGGGCCTGGGCAACATCATTACCCGACGGTTCGGCTCG' \
                     'GTCGACCTGGTCGCGTTGGTGATCTGGGGTGGGCTGATACCGCCACTTCCATTCCTGGCACTGTCCTGGTGGCTGGAAGGC' \
                     'CCCGAGCGCATTGGCCATGCCTTGGCCAATATCAGTTGGAGTTCGGTGCTGGCCCTGGCGTATTTGGCCTTTGTGGCCACC' \
                     'ATGCTCGGCTACAGCCTGTGGAGCAAGTTGCTGTCGCGTCACCCGGCAGGCAAAGTGGCGCCGTTCTCGCTGCTGGTACCG' \
                     'GTGATTGGCCTGAGTTCGTCGGCGTTGCTGCTGGGAGAGCGCCTGACCGCTACGCAGGGCTGGGGCGCATTGCTGGTGATG' \
                     'GCCGGCCTGCTGGTGAATGTATTCGGTGCGCGCATCGGTCAGCGTTTGCGGGCTGCCAACGCGTAAAGCGCTTAAATCTACC' \
                     'GAACTGCGCTCTGTTAGAATCGGCCTCTTTTTGCCAGGCCAACGGAGCGCATCCATGATCATTTCCACCACCAGCCAGCTCG' \
                     'AAGGCCGCCCGATTGCCGAATACCTGGGCGTGGTCAGTTCTGAATCGGTGCAGGGCATCTACTTCGTGCGCGATTTCTTTGC' \
                     'ACGGTTTCGCGACTTCTTCGGTGGCCGTTCGCAAACCCTGGAAAGCGCGCTGCGCGAGGCCCGGGAGCAGGCCACTGAAGAA' \
                     'CTCAAGGCCCGGGCACGACAGTTGCAAGCCGATGCGGTAGTGGGGGTGGATTTTGAGATCAGCATGCCGTCGGTACAGGGCG' \
                     'GCATGGTCGTGGTATTTGCCACCGGTACGGCAGTGCGCCTGAAGTAAAGCCCTTTGCCATTGGTCGGCTACAAGGTTCTTGT' \
                     'CCGGTCGGTCCGCAAATGACCAGCTAGTCTCACTTTCACAGGCCGCGTTTCCTGGGCAACTTGTCTGGTTGCCGACGCGACC' \
                     'GCCACTGGAGGGAGACTGGGCATGAGCGAATCCTTTTTCGAAGACCTGAACGATGCGTTCCCGATCAACAGCCAGGTGCGTT' \
                     'GCGGCCAGGCGGCATTTCGCCTGGGTTTCGCCCACATGACGCTGGATGATTCCGAACAGCTCCAGCCTGCACATTTGCAGCG' \
                     'CAGCAAAAAAGGCCGCTTCATGCCGCGGGTACCGCTCAAGAAGTGATCCCGCGTCTGCACACACCCCGCCCAACGGCGGGGT' \
                     'TTTTATTGCCATTCGTTGATCCGCCTCATGGTATTGCCCGGTAGCGCTCGCCGGATGGCCCGGCCTGTGGTTTTCTGACGCG' \
                     'GCATTCCAGCAACGGAGGATTGAATGCTCATGCGAGTCAGGGAAGAAACCTATTGGCAGTGGGCCGACGCACAGCTGCACAG' \
                     'CCGCTGCCACGACGAAGCACTCAGCGACGGCACCACACTGGACGTGCAGGTGCGCTTGTCGCGCCTGGGGGCGACGCAGCTG' \
                     'TTTCTGGGGTTGTACGCCGGGGATGGGCG'
G_DEFAULT_sgRNA = 'AGAACCTTGTAGCCGACCAA'
G_DEFAULT_PAM = 'TGG'

logger = logging.getLogger(__name__)

G_DICT_NUM_DNA_TABLE = {'Appending': 0, 'N': 1, 'Start': 2, 'End': 3, 'A': 4, 'T': 5, 'C': 6, 'G': 7, 'Special': 8, 'label_0': 9, 'label_1': 10, '+': 8}

def numSequence(in_listSequence: list, in_dictNumTable: dict = G_DICT_NUM_DNA_TABLE, in_intLength = G_INT_SEQUENCELENGTH):
    numberDNA = []
    for dna in in_listSequence:
        currentSeq = []
        for alphabet in dna:
            if alphabet not in in_dictNumTable:
                logger.warning(f"Error :: can't find current vocabulary {alphabet}")
                break
            else:
                currentSeq.append(in_dictNumTable[alphabet])
        if len(dna) == len(currentSeq):
            while len(currentSeq) < in_intLength:
                currentSeq.append(in_dictNumTable['Appending'])
            numberDNA.append(currentSeq)
        else:
            logger.warning(f"Loss sequence {dna}")

    return numberDNA

class buildCrispDataSet(Dataset):
    def __init__(self, promoter: list, sgRNA: list, pam: list):
        temSeq = []
        for index in range(len(promoter)):
            temSeq.append(promoter[index].strip().upper() + '+' + sgRNA[index].strip().upper() + '+' + pam[index].strip().upper())
        self.examples = numSequence(temSeq, G_DICT_NUM_DNA_TABLE, G_INT_SEQUENCELENGTH)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def loadDatasetFromFile(args):
    data = pd.read_excel(args.seq_file)
    promoter = data['Promoter'].tolist()
    sgRNA = data['sgRNA'].tolist()
    pam = data['PAM'].tolist()
    if 1 != len(set([len(promoter), len(sgRNA), len(pam)])):
        raise Exception(f"The gene editing system data read is missing, with {len(promoter)} promoter sequences, "
                        f"{len(sgRNA)} sgRNA sequences, and {len(pam)} PAM sequences. Please check.")
    for index in range(len(promoter)):
        if not checkSeq(promoter[index]):
            raise Exception(f"Please check the input promoter sequence{promoter[index]}, which contains invalid nucleotides.")
        if not checkSeq(sgRNA[index]):
            raise Exception(f"Please check the input sgRNA sequence{sgRNA[index]}, which contains invalid nucleotides.")
        if not checkSeq(pam[index]):
            raise Exception(f"Please check the input PAM sequence{pam[index]}, which contains invalid nucleotides.")

    return buildCrispDataSet(promoter, sgRNA, pam)


def checkSeq(seq: str, legalBase = ['A', 'T', 'C', 'G']):
    for base in seq.upper():
        if base not in legalBase:
            logger.warning(f"[checkSeq] Warnning:: illegal base {base}")
            return False
    return True

def tokenizeCrispSystem(args):
    promoter, sgRNA, pam = args.promoter, args.sgRNA, args.pam
    if not checkSeq(promoter):
        raise Exception(f"Please check the input promoter sequence{promoter}, which contains invalid nucleotides.")
    if not checkSeq(sgRNA):
        raise Exception(f"Please check the input sgRNA sequence{sgRNA}, which contains invalid nucleotides.")
    if not checkSeq(pam):
        raise Exception(f"Please check the input PAM sequence{pam}, which contains invalid nucleotides.")
    synSeq = promoter.strip().upper() + '+' + sgRNA.strip().upper() + '+' + pam.strip().upper()
    numSeq = numSequence([synSeq], G_DICT_NUM_DNA_TABLE, G_INT_SEQUENCELENGTH)

    return torch.tensor(numSeq, dtype=torch.long)


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

# 测试集文件保存格式为：evalRecording+'EpochNum'.txt
def getFileNum(in_strFolder: str, in_strPrefix: str = 'evalRecording'):
    t_listDirAndFile = os.listdir(in_strFolder)
    t_listFiles = []
    for file in t_listDirAndFile:
        if os.path.isfile(os.path.join(in_strFolder, file)):
            t_listFiles.append(os.path.splitext(file)[0])  # delete file extension
    num = []
    for file in t_listFiles:
        if file.startswith(in_strPrefix):
            num.append(int(file[13:]))  # evalRecording共13个字符
    if 0 < len(num):  # folder has not recording
        return max(num) + 1  # add 1 to file index
    else:
        return 1  # file index is 1 when folder is vancant

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

def singleCrispPrediction(args, model: PreTrainedModel):
    os.makedirs(args.output_dir, exist_ok=True)
    numSequence = tokenizeCrispSystem(args)
    inputs = numSequence.clone().detach().to(args.device)

    with torch.no_grad():    # 不进行梯度跟踪
        outputs = model(inputs, labels=torch.tensor([0], dtype=torch.long))
        _, logits = outputs[:2]
        pred = logits.tolist()
        # record label, prediction and loss in this batch by log
        deSequences = decoderSeq(inputs.tolist(), G_DICT_NUM_DNA_TABLE)
        logger.info(f"Eval:: input: {deSequences[0]}")
        logger.info(f'Eval:: Prediction: {pred}')
        outFileName = 'predictRecord.txt'
        with open(os.path.join(args.output_dir, outFileName), 'a+') as out:
            if len(deSequences) != len(pred):
                logger.info("Error:: can't record sequence and prediction with different size")
            for index in range(len(deSequences)):
                out.write(deSequences[index] + '\t' + str(pred[index][0]) + '\n')

def mutilCrispPrediction(args, model: PreTrainedModel):
    os.makedirs(args.output_dir, exist_ok=True)
    predDataset = loadDatasetFromFile(args)

    # prepare for dataset
    sampler = SequentialSampler(predDataset)
    eval_dataloader = DataLoader(
        predDataset, sampler=sampler, batch_size=1
    )
    # Eval!
    model.eval()
    allPrediction = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch.clone().detach().to(args.device)
        with torch.no_grad():    # 不进行梯度跟踪
            outputs = model(inputs, labels=torch.tensor([0], dtype=torch.long))
            _, logits = outputs[:2]
            for pred in logits.tolist():
                for subPred in pred:
                    allPrediction.append(subPred)
            # record label, prediction and loss in this batch by log
            deSequences = decoderSeq(inputs.tolist(), G_DICT_NUM_DNA_TABLE)
            outFileName = Path(args.seq_file).stem + '_predRecord.txt'
            with open(os.path.join(args.output_dir, outFileName), 'a+') as out:
                pred = logits.tolist()
                if len(deSequences) != len(pred):
                    logger.info("Error:: can't record sequence and prediction with different count")
                for index in range(len(deSequences)):
                    out.write(deSequences[index] + '\t' + str(pred[index][0]) + '\n')

def main(args):
    # Setup CUDA, GPU
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, "Console-output.log"),
        filemode='w'
    )

    # Load model weights based on parameter -- model_path
    if args.model_path:
        llamaConfiguration = llama.LlamaConfig.from_pretrained(args.model_path, cache_dir=None)
        # llamaConfiguration.num_labels = 1
        # llamaConfiguration.pad_token_id = 0
        model = llama.LlamaForSequenceClassification.from_pretrained(
            args.model_path,
            from_tf=bool(".ckpt" in args.model_path),
            config=llamaConfiguration,
            cache_dir=None,
        )
        model.to(args.device)
    else:
        raise Exception(f"During prediction, the parameter '--model_path' must be provided.")

    singleCrispPrediction(args, model)
    if args.seq_file is not None:
        mutilCrispPrediction(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The model checkpoint for prediction.",
    )
    parser.add_argument(
        "--promoter", default=G_DEFAULT_PROMOTER, type=str, help="The promoter sequence that needs to be predicted."
    )
    parser.add_argument(
        "--sgRNA", default=G_DEFAULT_sgRNA, type=str, help="The sgRNA sequence that needs to be predicted."
    )
    parser.add_argument(
        "--pam", default=G_DEFAULT_PAM, type=str, help="The PAM sequence that needs to be predicted."
    )
    parser.add_argument(
        "--seq_file", default=None, type=str, help="Crisp sequence file that needs to be predicted."
    )

    args = parser.parse_args()

    main(args)
