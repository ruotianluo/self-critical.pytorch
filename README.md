# Attention on Attention for Image Captioning

This repository includes the implementation for [Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954).

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.0
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule)
- tensorboardX


## Training AoANet

### Prepare data

See details in `data/README.md`.

(**notes:** Set `word_count_threshold` in `scripts/prepro_labels.py` to 4 to generate a vocabulary of size 10,369.)

You should also preprocess the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

```bash
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```
### Start training

```bash
$ CUDA_VISIBLE_DEVICES=0 sh train.sh
```

See `opts.py` for the options. (You can download the pretrained models from [here](https://drive.google.com/drive/folders/1ab0iPNyxdVm79ml-oozsIlH7H6t6dIVl?usp=sharing).)


### Evaluation

```bash
$ CUDA_VISIBLE_DEVICES=0 python eval.py --model log/log_aoanet_rl/model.pth --infos_path log/log_aoanet_rl/infos_aoanet.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test
```

### Performance
You will get the scores close to below after training under xe loss for 25 epochs:
```python
{'Bleu_1': 0.7729384559899702, 'Bleu_2': 0.6163398035383025, 'Bleu_3': 0.4790123137715982, 'Bleu_4': 0.36944349063530374, 'METEOR': 0.2848188431924821, 'ROUGE_L': 0.5729849683867054, 'CIDEr': 1.1842173801790759, 'SPICE': 0.21650786258302354}
```
(**notes:** You can enlarge `--max_epochs` in `train.sh` to train the model for more epochs and improve the scores.)

after training under SCST loss for another 15 epochs, you will get:
```python
{'Bleu_1': 0.8054903453672397, 'Bleu_2': 0.6523038976984842, 'Bleu_3': 0.5096621263772566, 'Bleu_4': 0.39140307771618477, 'METEOR': 0.29011216375635934, 'ROUGE_L': 0.5890369750273199, 'CIDEr': 1.2892294296245852, 'SPICE': 0.22680092759866174}
```


## Reference

If you find this repo helpful, please consider citing:

```
@inproceedings{huang2019attention,
  title={Attention on Attention for Image Captioning},
  author={Huang, Lun and Wang, Wenmin and Chen, Jie and Wei, Xiao-Yong},
  booktitle={International Conference on Computer Vision},
  year={2019}
}
```

## Acknowledgements

This repository is based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch), and you may refer to it for more details about the code.
