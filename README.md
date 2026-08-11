## Introduction
> - GEO-SeqGuider is a genomic language model developed to predict CRISPR regulatory activity in Pseudomonas putida. Built on the Llama architecture, the model was trained through a two-stage process that first incorporates large-scale bacterial genomic sequences and then specializes in CRISPR system parameters, including promoter regions, sgRNA sequences, and PAM motifs. GEO-SeqGuider takes these sequence inputs and quantitatively predicts the resulting gene expression levels under CRISPR activation or interference conditions. By learning the complex sequence-activity relationships, the model enables in silico screening of potential regulatory targets, significantly reducing the experimental workload required for CRISPRa/i design. It serves as a key computational component of our multifunctional CRISPR-AI(D) platform. Ultimately, GEO-SeqGuider enhances the precision and predictability of gene regulation, supporting efficient metabolic engineering in P. putida.
> - We describe GEO-SeqGuider in our paper[XXX](https://doi.org/XXXX)

## Important links:  
- [Paper:XXXX](https://arxiv.org/abs/XXXXXXX)
- [Huggingface: model weights](https://huggingface.co/lileica/GEO-SeqGuider)
- [Dataset: The GEO-SeqGuider model is trained in two stages: genome pre training and specific task fine-tuning, based on the dataset ]( https://doi.org/XXXX/zenodo.XXX)

## Installation with conda
```bash
    conda env create -f requirements.yml
    conda activate GEOSeqGuiderEnv
```

## Applications

### pretiction step by step

#### step 1
Download model weights from huggingface
```bash
git clone https://huggingface.co/lileica/GEO-SeqGuider localModel
```

#### step 2
Predict with local model weights
```bash
    python GEOSeqGuider.py --model=localModel/foldx --promoter="XXXX" --sgRNA='XXXX' --pam='XXXX' --out_dir='XXXX'
```


# Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{XXX,
  author  = {XXX,XXX,XXX},
  journal = {CNS},
  title   = {XXXXXXXXXXX},
  year    = {2026},
  volume  = {14},
  URL={https://www.XXXX},       
	DOI={XXX.2026.XXX},      
	ISSN={XXXX-XXXX}
}
```
