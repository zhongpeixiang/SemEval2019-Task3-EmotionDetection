## SemEval 2019 Task 3 - Emotion Detection in Textual Conversations

### Dependencies
- Python 3.5+
- Numpy 1.15+
- PyTorch 1.0+
- torchtext 0.3.1+
- Spacy 2.0+

### Model
- Recurrent Convolutional Neural Network (RCNN) (Lai 2015, AAAI)

### External Word Representations
- Word embedding from datastories: https://github.com/cbaziotis/datastories-semeval2017-task4
- Word emmedding from trained sentiment classifier using CNN (Kim 2014, EMNLP), follow https://github.com/cbaziotis/datastories-semeval2017-task4
- ELMo: https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
- BERT: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py

### External Sentence Representations:
- DeepMoji: https://github.com/huggingface/torchMoji/blob/master/examples/encode_texts.py
- InferSent: https://github.com/facebookresearch/InferSent/blob/master/encoder/demo.ipynb

### Run:
- Set your hyper-parameters in train.json, each key can have multiple values for grid search
- All word and sentence embeddings are in numpy array of shape (num_examples, embedding_size)
- Run `python train.py --config train.json`

### Credit:
If you find the code or paper useful, please cite:
> @article{zhong2019ntuer,
  title={ntuer at SemEval-2019 Task 3: Emotion Classification with Word and Sentence Representations in RCNN},
  author={Zhong, Peixiang and Miao, Chunyan},
  journal={arXiv preprint arXiv:1902.07867},
  year={2019}
}