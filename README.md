# Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query [EMNLP 2025]


This is the repo for our EMNLP 2025 paper:
[Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query](https://arxiv.org/abs/2505.20334)


## Acknowledgement

Our codebase is built upon **[KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)**.  
We sincerely thank the authors for providing open-source code to support this project.  



## News

- [2025-8-21] 🎉🎉🎉 Our paper has been accepted to **EMNLP 2025 Main Conference**!

## Installation

```bash

git clone git@github.com:noforit/Lookahead_Q-Cache.git
cd Lookahead_Q-Cache
conda create -n LAQ python=3.12
conda activate LAQ
pip install -r requirements.txt .

```
## LongBench

You can obtain the results of LAQ on LongBench by referring to the `run_LAQ.sh` and modifying the corresponding parameters.

```bash
bash run_LAQ.sh
```


## Needle in haystack


```bash
bash run_needle.sh
```

After inference, run

`python scripts/scripts_needle/visualize.py`

to draw the img, you should change `FOLDER_PATH` in `visualize.py` to your output path.


## Citation

If you find **Lookahead Q-Cache** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{wang2025lookahead,
  title={Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query},
  author={Wang, Yixuan and Ji, Shiyu and Liu, Yijun and Xu, Yuzhuang and Xu, Yang and Zhu, Qingfu and Che, Wanxiang},
  journal={arXiv preprint arXiv:2505.20334},
  year={2025}
}
```

