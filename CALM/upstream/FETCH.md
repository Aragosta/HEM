# Fetching the parts not vendored here

`models/` and `train/` are vendored verbatim (MIT, see `LICENSE`). The tokenizer
(11 MB), the validation set and `overview.png` are not, to keep this repository
lean. To get the complete upstream tree:

```bash
git clone https://github.com/shaochenze/calm.git
```

Training data (~2.5 TB) comes from `data/get_data.sh` in that clone.
Pretrained checkpoints are on Hugging Face under `cccczshao/`.
