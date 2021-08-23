
<div align="center">

#  Normalizing Flows for Fast Detector Simulation
 
This repository contains code for Fast Detector Simulation using Normalizing Flows, specifically [Block Neural Autoregressive Flow](https://arxiv.org/abs/1904.04676) 
and [Masked Autoregressive Flows](https://proceedings.neurips.cc/paper/2017/file/6c1da886822c67822bcf3679d04369fa-Paper.pdf). This work was done as part of GSoC '21.

<a href="https://ml4sci.org/" target="_blank"><img alt="gsoc@ml4sci" height="200px" src="https://github.com/Anantha-Rao12/Decoding-Quantum-States-with-NMR/blob/main/Assests/gsoc-ml4sci.jpg" /></a> 
</div>


# Table of Contents

- [Getting Started](#getting-started)
- [Directory Structure](#directory-structure)
- [Colab Links](#colab-links)
- [Results](#results)
    + [Training using BNAF](#training-using-bnaf)
    + [Training using MAF](#training-using-maf)
    + [Results from BNAF](#results-from-bnaf)
    + [Results from MAF](#results-from-maf)
- [Model Configs](#model-configs)
- [Resources and References](#resources-and-references)
    + [Drive Links](#drive-links)
    + [Implementations](#implementations)
    + [Blogs and Tutorials](#blogs-and-tutorials)
- [Contact](#contact)
- [Cite](#cite)



# Getting Started
- Weights and Configs for trained models can be found in the `results` directory. 
- [Weights and Biases](https://wandb.ai/site) was used to track the training progress. The corresponding config files can be found in `logs/wandb`. 
- Notebooks to rerun the experiments can be found in the `nbs` directory. 
- The dataset can be downloaded from this [link](https://cernbox.cern.ch/index.php/s/xcBgv3Vw3rmHu9u) using the `get_dataset.py` script. 


# Directory Structure
```
Falcon-NF
├── data                                     # Download Link Below
│   ├── Boosted_Jets_Sample-0.snappy.parquet
│   ├── Boosted_Jets_Sample-1.snappy.parquet
│   ├── Boosted_Jets_Sample-2.snappy.parquet
│   ├── Boosted_Jets_Sample-3.snappy.parquet
│   └── Boosted_Jets_Sample-4.snappy.parquet
├── results                                  # Available on Drive
│   └── run_
│       ├── 2021-06-18_11-20-46
│       ├── 2021-06-18_11-20-46
│       ..  
│       └── 2021-07-14_10-07-06
├── logs                                     # Available on Drive
│   └── wandb
│       ├── run-20210625_104225-jky52l62
│       ├── run-20210625_104619-1yqfgb06
│       ..  
│       └── latest-run 
├── nbs
│   ├── BNAF_1.ipynb
│   ├── BNAF_2.ipynb
│   ├── BNAF_3.ipynb
│   ├── BNAF_4.ipynb
│   └── MAF.ipynb
└── get_dataset.py

```
Note: Temporary files have been excluded.




# Colab Links

It is recommended to use Google Colab to view/run the below notebooks. 

- BNAF 1 - [Link](https://github.com/adiah80/Falcon-NF/blob/main/nbs/BNAF_1.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiah80/Falcon-NF/blob/main/nbs/BNAF_1.ipynb)
- BNAF 2 - [Link](https://github.com/adiah80/Falcon-NF/blob/main/nbs/BNAF_2.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiah80/Falcon-NF/blob/main/nbs/BNAF_2.ipynb)
- BNAF 3 - [Link](https://github.com/adiah80/Falcon-NF/blob/main/nbs/BNAF_3.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiah80/Falcon-NF/blob/main/nbs/BNAF_3.ipynb)
- BNAF 4 - [Link](https://github.com/adiah80/Falcon-NF/blob/main/nbs/BNAF_4.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiah80/Falcon-NF/blob/main/nbs/BNAF_4.ipynb)
- MAF - [Link](https://github.com/adiah80/Falcon-NF/blob/main/nbs/MAF.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiah80/Falcon-NF/blob/main/nbs/MAF.ipynb)




# Results

### Training using BNAF
![image](https://user-images.githubusercontent.com/34454784/125622473-9f27da0f-55e5-4817-835e-80800336d5c4.png)

### Training using MAF
<img width="1212" alt="MAF Training" src="https://user-images.githubusercontent.com/34454784/130432481-d1e9a9a5-1a5b-4d36-a6b9-4a5290bbafff.png">

### Results from BNAF
<img width="1111" alt="BNAF Results" src="https://user-images.githubusercontent.com/34454784/130431982-9421c40a-8798-4cd2-aff5-90f185574f79.png">

### Results from MAF
<table>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/34454784/130430351-78d17442-e633-48d3-b348-521277014144.png" alt="1" width = 480px height = 320px </td>
    <td><img src="https://user-images.githubusercontent.com/34454784/130430363-d3d32698-a653-4ed0-b5ea-c1e35fb53c8e.png" alt="2" width = 480px height = 320px></td>
  </tr> 
  <tr>
    <td><img src="https://user-images.githubusercontent.com/34454784/130430370-a798f138-23da-463c-95d2-82e18dd84880.png" alt="3" width = 480px height = 320px></td>
    <td><img src="https://user-images.githubusercontent.com/34454784/130430373-dc7afd7c-9b79-4010-ab46-93818f6074c5.png" alt="4" width = 480px height = 320px></td>
  </tr>
</table>



# Model Configs

Detailed Model Configs can be found in `results/run_/<checkpoint_ID>/config.txt`.
```
Parsed args:
{'batch_size': 16,
 'cuda': 0,
 'data_dim': 1536,
 'device': device(type='cpu'),
 'hidden_dim': 3072,
 'log_interval': 200,
 'lr': 0.1,
 'lr_decay': 0.5,
 'lr_patience': 2000,
 'n_hidden': 3,
 'n_steps': 1000,
 'output_dir': './results/run_/2021-07-14_10-07-06',
 'plot': True,
 'pt_range': [50, 55],
 'restore_file': None,
 'seed': 0,
 'step': 0,
 'train': False}

Num trainable params: 37,776,384

Model:
BNAF(
  (net): FlowSequential(
    (0): MaskedLinear(in_features=1536, out_features=3072, bias=True)
    (1): Tanh()
    (2): MaskedLinear(in_features=3072, out_features=3072, bias=True)
    (3): Tanh()
    (4): MaskedLinear(in_features=3072, out_features=3072, bias=True)
    (5): Tanh()
    (6): MaskedLinear(in_features=3072, out_features=3072, bias=True)
    (7): Tanh()
    (8): MaskedLinear(in_features=3072, out_features=1536, bias=True)
  )
)
```


# Resources and References

### Drive Links
- Dataset: [Link](https://cernbox.cern.ch/index.php/s/xcBgv3Vw3rmHu9u)
- Logs : [Drive Link](https://drive.google.com/drive/folders/1-0703HEm-GsAYJOsYdr6-ebYSAw_rejp?usp=sharing)
- Results: [Drive Link](https://drive.google.com/drive/folders/1F_ENUHkIqCZnLxawwzhuGX7AWNBwhWAB?usp=sharing)

### Implementations
- Original BNAF implementation - [Link](https://github.com/nicola-decao/BNAF)
- Original MAF implementation - [Link](https://github.com/gpapamak/maf)
- Various NF implementations - [Link](https://github.com/kamenbliznashki/normalizing_flows)
- Previous GSoC Work by [Ali Harari](https://github.com/ahariri13) - [Link](https://github.com/ahariri13/FALCON)

### Blogs and Tutorials
- ICML Tutorial on Normalizing Flows by Eric Jang: [Link](https://slideslive.com/38917907/tutorial-on-normalizing-flows)
- Blog Posts by Eric Jang : [Blog 1](https://blog.evjang.com/2018/01/nf1.html), [Blog 2](https://blog.evjang.com/2018/01/nf2.html)
- Blog Post by Lilian Weng : [Link](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
- Blog Post by Emiel Hoogeboom: [Link](https://ehoogeboom.github.io/post/en_flows/)
- ECCV Tutorial by Marcus Brubaker: [Link](https://www.youtube.com/watch?v=u3vVyFVU_lI)
- CVPR Tutorial by Marcus Brubaker: [Link](https://www.youtube.com/watch?v=8XufsgG066A)
- PyMC3's Normalizing Flows Overview: [Link](https://docs.pymc.io/notebooks/normalizing_flows_overview.html)

# Contact
- Please contact [Aditya Ahuja](https://github.com/adiah80) for questions/comments related to this repository. 
- This project was mentored by [Prof. Sergei V. Gleyzer](http://sergeigleyzer.com/), [Prof. Harrison B. Prosper](http://www.hep.fsu.edu/~harry/) and [Prof. Michelle Kuchera](https://www.davidson.edu/people/michelle-kuchera).


# Cite

```
@misc{hariri2021graph,
      title={Graph Generative Models for Fast Detector Simulations in High Energy Physics}, 
      author={Ali Hariri and Darya Dyachkova and Sergei Gleyzer},
      year={2021},
      eprint={2104.01725},
      archivePrefix={arXiv},
      primaryClass={hep-ex}
}
```

```
@article{bnaf19,
  title={Block Neural Autoregressive Flow},
  author={De Cao, Nicola and
          Titov, Ivan and
          Aziz, Wilker},
  journal={35th Conference on Uncertainty in Artificial Intelligence (UAI19)},
  year={2019}
}
```

```
@misc{papamakarios2018masked,
      title={Masked Autoregressive Flow for Density Estimation}, 
      author={George Papamakarios and Theo Pavlakou and Iain Murray},
      year={2018},
      eprint={1705.07057},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
