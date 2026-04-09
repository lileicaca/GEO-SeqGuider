## 简介
> - 本仓库实现了GEO-SeqGuider模型的搭建与训练、预测。GEO-SeqGuider 是一款专为基因编辑系统效果预测而设计的高级软件。基因编辑技术在现代生物学研究和应用中扮演着至关重要的角色，但其复杂性也带来了诸多挑战。基因编辑系统涉及复杂的基因、调控元件（如启动子、增强子等）以及引导RNA的相关信息。为了应对这些挑战，GEO-SeqGuider 软件致力于将这些复杂的信息整合在一起，并基于大语言模型训练出一个能够有效预测基因编辑效果的模型。
> - 主要入口：`run_main.py`。它通过命令行参数获取模型训练相关配置，并依据这些参数从指定位置读取模型结构的默认配置。随后，该脚本调用 transformers 库的 API 完成 GEOSeqGuider 模型的初始化，并根据参数设置从 input 目录加载经过分词处理的基因组数据或基因编辑系统数据进行训练。训练过程中，程序将依据用户指定的参数执行模型学习，并将训练日志、训练好的模型及相关性能统计结果保存至 output 目录。
> - 数据存储：建议将原始数据统一存放于data目录下，经分词（tokenization）处理后的数据则存储于input目录。在模型运行时，用户需通过指定参数（如--train_data_file、--train_label_file、--eval_data_file、--eval_label_file等）传入相应数据文件的相对路径及文件名。该目录结构设计有助于实现数据资源的统一规范管理。
> - 结果与日志写入: 所有输出结果将根据用户指定的 output_dir 参数，保存至对应文件夹中。建议将该参数设置为 output 目录下的子文件夹。由于模型训练包含预训练和微调两个阶段，推荐在 output 下分别建立子文件夹以区分这两个阶段的输出。用户也可直接在 --output_dir 参数中指定子文件夹路径，模型在运行时会自动创建相应目录。
训练过程中，相关日志将以 Console-output.log 为文件名保存至输出目录；模型权重则会以 checkpoint-XXX 为名创建子文件夹进行存储。此外，训练期间的损失值（loss）、困惑度（ppl）、token 数量等信息也会以文件形式保存至输出目录。
上述数据同时已通过 TensorBoard 进行可视化，相关日志保存在输出目录下的 SummaryWriter 子文件夹中。用户只需在输出目录中执行以下命令：

```bash
    tensorboard --logdir=SummaryWriter
```

> 随后按提示在浏览器中打开相应地址，即可查看可视化结果。

## 支持操作系统
> - Linux(Ubuntu 11.4.0-1ubuntu1~22.04)

## 硬件配置
> - CPU 20核心
> - 内存 629G
> - A100-PCIE-40GB(驱动版本：535.261.03，CUDA版本：12.2)

## 主要文件说明
> - `run_main.py` — 主程序：训练模型（包括学习基因组和基因编辑系统的数据集）
> - `tokenize_crisp.py` — 基因编辑系统数据的tokenization。
> - `tokenize_Genome.py` — 基因组数据集的tokenization。

## 执行指导与示例
> - Step1 创建虚拟环境并且激活
```bash
    conda env create -f environment.yml
    conda activate GEOSeqGuiderEnv
```

> - Step2 基因组tokenization： 用户需要将基因组文件提前放入data目录下，并且在安装了依赖包的python环境下执行以下命令

```bash
    python tokenize_Genome.py --genome=data/kt2440.fa --output_dir=input
```

> - Step3 模型预训练：经过上一步的基因组tokenization之后，在目录input下面会生成两个文件train_numbase_2048.jsonl与test_numbase_2048.jsonl,用户需要将这两个文件作为模型预训练的参数，在安装了依赖包的python环境下执行以下命令
```bash
    python run_main.py --output_dir=output/pretrain \
                    --train_data_file=input/train_numbase_2048.jsonl \
                    --eval_data_file=input/test_numbase_2048.jsonl \
                    --model_config_file=input/config.json \
                    --do_train \
                    --model_task 0 \
                    --gradient_accumulation_steps 10 \
                    --num_train_epochs 1 \
                    --per_gpu_train_batch_size 5 \
                    --per_gpu_eval_batch_size 5 \
                    --save_steps 10 \
                    --save_total_limit 2 \
                    --evaluate_during_training \
                    --logging_steps 1 \
                    --line_by_line \
                    --learning_rate 4e-4 \
                    --block_size 2048 \
                    --adam_epsilon 1e-6 \
                    --weight_decay 0.01 \
                    --beta1 0.9 \
                    --beta2 0.98 \
                    --warmup_steps 18700 \
                    --overwrite_output_dir
```

> 注意，程序运行的所有参数在脚本run_main.py有详细的定义，如果用户使用其学习其他物种的基因组，需要根据其定义以及实际运行的计算资源进行调整。

> - Step4 Crisp数据集tokenization： 用户需要将实验收集的Crisp数据集提前放入data目录下，并且在安装了依赖包的python环境下执行以下命令

```bash
    python tokenize_crisp.py --data=data/data.xlsx --output_dir=input
```

> - Step5 模型微调：经过上一步的Crisp数据集tokenization之后，在目录input下面会生成四个文件training_gene.jsonl、trainingLab_gene.json、test_gene.jsonl、testLab_gene.json,用户需要将这几个文件作为模型微调的训练集、训练集标签、测试集、测试集标签，在安装了依赖包的python环境下执行以下命令
```bash
    python run_main.py --output_dir=output/finetune \
                    --model_name_or_path=output/pretrain/checkpoint-XXX \
                    --train_data_file=input/train_numbase_2048.jsonl \
                    --train_label_file=input/trainingLab_gene.json \
                    --eval_data_file=input/test_gene.jsonl \
                    --eval_label_file=input/testLab_gene.json \
                    --do_train \
                    --model_task 1 \
                    --gradient_accumulation_steps 10 \
                    --num_train_epochs 1 \
                    --per_gpu_train_batch_size 5 \
                    --per_gpu_eval_batch_size 5 \
                    --save_steps 10 \
                    --save_total_limit 2 \
                    --evaluate_during_training \
                    --logging_steps 1 \
                    --line_by_line \
                    --learning_rate 4e-4 \
                    --block_size 2048 \
                    --adam_epsilon 1e-6 \
                    --weight_decay 0.01 \
                    --beta1 0.9 \
                    --beta2 0.98 \
                    --warmup_steps 18700 \
                    --overwrite_output_dir
```
> 请注意：model_name_or_path 参数应设置为预训练输出的 checkpoint 文件夹（推荐选用编号最大者）。其余运行参数详见 run_main.py 中的定义。

> - Step6 crisp数据集预测：经过上述步骤的训练，模型已能够基于CRISPR系统预测基因表达量。用户需要在已安装相应依赖包的 Python 环境中运行以下命令，进行数据tokenization和预测。
```bash
    #### 对data.xlsx的 CRISP 数据集进行tokenization
    python tokenize_crisp.py --data=data/data.xlsx --outputDir=input ----prediction=1
    #### 对经过 tokenization 的 CRISP 数据集进行预测
    python run_main.py --output_dir=output \
                    --model_name_or_path=output/pretrain/checkpoint-XXX \
                    --eval_data_file=input/predict_gene.jsonl

```
> 请注意：
> 1. model_name_or_path: 指定用于预测的模型路径。该路径应指向 Step5 输出目录下的 checkpoint 文件夹。建议用户选择编号最大的文件夹，代表训练步数最多的模型。此外，我们也提供了一个预训练模型（位于 model 目录），用户可通过将此参数设置为 model 来直接调用。
> 2. eval_data_file: 指定待预测数据的文件路径。该文件必须是已经过 tokenization 处理的 CRISPR 数据集。
> 3. 对于不需要训练模型的用户，可以直接跳过Step1 - Step5，使用model里面保存的模型权重进行预测。


## 程序中的一些约定
> - 在tokenization过程中，每条数据均为基因序列，其元素仅由A、T、C、G及‘+’五种字符组成。处理后的序列统一以JSON Lines格式保存，而CRISPR数据集的标签则统一存储于JSON文件中。
> - 输出目录命名规则：为便于管理，输出目录 (output_dir) 遵循以下命名与存储规则
>  1. checkpoint-XXXX：用于保存训练过程中的历史模型检查点。
>  2. epochModel：在微调阶段，每个epoch训练结束后的模型会单独保存于此。
>  3. historyRecord：记录每个epoch结束后的模型预测结果及性能表现。
>  4. SummaryWriter：存放TensorBoard可视化日志，内容包括训练集/测试集的损失、测试集的困惑度及微调模型表现等。

## 联系人及联系方式
> - 姓名：李磊
> - 电话： 18351806287       微信：DL1046531516
