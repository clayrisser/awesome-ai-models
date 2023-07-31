# awesome-ai-models

> a curated list of awesome ai models

# TOPOLOGY: Transformers

## NLP (Natural Language Processing)

### baichuan-vicuna-7b.ggmlv3.q4_0

Baichuan-7B is an open-source large-scale pre-trained model developed by Baichuan Intelligent Technology. Based on the Transformer architecture, it is a model with 7 billion parameters trained on approximately 1.2 trillion tokens. It supports both Chinese and English, with a context window length of 4096.

#### STEPS:

Download the model-> WGET https://huggingface.co/TheBloke/baichuan-vicuna-7B-GGML/resolve/main/baichuan-vicuna-7b.ggmlv3.q4_0.bin -O models/baichuan-vicuna-7b.ggmlv3.q4_0

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "baichuan-vicuna-7b.ggmlv3.q4_0",
"messages": [{"role": "user", "content": "i need a coffee for"}],
"temperature": 0.9
}'

- [ ] rest api
- [ ] cpu
- [ ] Completed

## ggml-gpt4all-j

A commercially licensable model based on GPT-J and trained by Nomic AI on the v0 GPT4All dataset.

The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.

#### STEPS:

Download the model-> wget https://gpt4all.io/models/ggml-gpt4all-j.bin -O models/ggml-gpt4all-j.bin

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "ggml-gpt4all-j",
"messages": [{"role": "user", "content": "How are you?"}],
"temperature": 0.9
}'

- [ ] rest api
- [ ] cpu
- [ ] Completed

# WEB TOPOLOGY

## NLP (natural language processing)

### Experience: Fill Mask

#### Models: [ bert-base-uncased, roberta-base, roberta-large ]

#### Description:

an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives

### Experience: Question Answering

### Models: [ alphakavi22772023/bertQA, timpal0l/mdeberta-v3-base-squad2, LLukas22/all-MiniLM-L12-v2-qa-en ]

#### Description:

All these datasets were concatenated into a single dataset that we called frenchQA.

### Experience: Sentence Similarity

### Models: [ intfloat/e5-large-v2, jinaai/jina-embedding-s-en-v1, flax-sentence-embeddings/all_datasets_v4_MiniLM-L6 ]

#### Description

the model should predict which out of a set of randomly sampled other sentences, was actually paired with it in our dataset

### Experience:Summarization

### Models:[ facebook/bart-large-cnn, csebuetnlp/mT5_multilingual_XLSum, slauw87/bart_summarisation ]

#### Description

BigBird relies on block sparse attention instead of normal attention (i.e. BERT's attention) and can handle sequences up to a length of 4096 at a much lower compute cost compared to BERT

### Experience:Text Classification

### Models:[ SamLowe/roberta-base-go_emotions, ProsusAI/finbert, nlptown/bert-base-multilingual-uncased-sentiment]

#### Description:

This model is intended for direct use as a sentiment analysis model for product reviews in any of the six languages above or for further finetuning on related sentiment analysis tasks

### Experience:Text Generation

### Models:[ gpt2, HuggingFaceH4/starchat-beta, gpt2-xl, fxmarty/tiny-testing-gpt2-remote-code]

#### Description:

The model is a pretrained model on English language using a causal language modeling (CLM) objective.

### Experience:Text-to-text

### Models:[ humarin/chatgpt_paraphraser_on_T5_base, juierror/flan-t5-text2sql-with-schema,voidful/context-only-question-generator ]

#### Description

This model is a sequence-to-sequence question generator which takes context as an input, and generates a question as an output.

### Experience:Token Classification

### Models:[ dslim/bert-base-NER, DOOGLAK/01_SR_500v9_NER_Model_3Epochs_AUGMENTED, DOOGLAK/Paraphrased_500v3_NER_Model_3Epochs_AUGMENTED]

#### Description

A Named Entity Recognition model trained on a customer feedback data using DistilBert.

### Experience: Translation

### Models:[ Helsinki-NLP/opus-mt-fi-mt,t5-base, Helsinki-NLP/opus-mt-fr-tvl, Helsinki-NLP/opus-mt-eu-ru ]

#### Description

Our text-to-text framework allows us to use the same model, loss function, and hyperparameters on any NLP task

### Experience:Zero-Shot Classification

### Models:[ MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, Narsil/deberta-large-mnli-zero-cls, HiTZ/A2T_RoBERTa_SMFA_TACRED-re]

#### Description

This model is the Multi-Genre Natural Language Inference (MNLI) fine-turned version of the uncased

## Computer Vision

### Experience:Image Classification

### Models:[ google/vit-base-patch16-224, microsoft/swinv2-tiny-patch4-window8-256, victor/autotrain-satellite-image-classification-40975105875 ]

#### Description

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks

### Experience: Image Segmentation

### Models:[ mattmdjaga/segformer_b2_clothes, nvidia/segformer-b3-finetuned-ade-512-512, zoheb/mit-b5-finetuned-sidewalk-semantic ]

#### Description

SegFormer consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great results on semantic segmentation benchmarks such as ADE20K and Cityscapes

### Experience:Object Detection

### Models:[ facebook/detr-resnet-50, hustvl/yolos-tiny, facebook/detr-resnet-50-dc5, facebook/detr-resnet-101]

#### Description

The DETR model is an encoder-decoder transformer with a convolutional backbone. Two heads are added on top of the decoder outputs in order to perform object detection

## Audio

### Experience:Automatic Speech Recognition

### Models:[ openai/whisper-tiny.en, m3hrdadfi/wav2vec2-large-xlsr-persian-v3, facebook/s2t-medium-librispeech-asr ]

#### Description:

S2T is an end-to-end sequence-to-sequence transformer model. It is trained with standard autoregressive cross-entropy loss and generates the transcripts autoregressively

## Multimodal

### Experience:Feature Extraction

### Models:[ facebook/bart-large, cambridgeltl/SapBERT-from-PubMedBERT-fulltext, DeepPavlov/rubert-base-cased-sentence ]

#### Description:

The Vision Transformer (ViT) is a transformer encoder model (BERT-like). Images are presented to the model as a sequence of fixed-size patches

### Experience:Image-to-Text

### Models:[ nlpconnect/vit-gpt2-image-captioning, microsoft/trocr-small-printed, naver-clova-ix/donut-base-finetuned-cord-v1 ]

#### Description:

GIT is a Transformer decoder conditioned on both CLIP image tokens and text tokens

### Experience:Zero-Shot Image Classification

### Models:[ openai/clip-vit-large-patch14, laion/CLIP-ViT-H-14-laion2B-s32B-b79K, laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K ]

#### Description:

Explore an alternative to ViT and ResNet (w/ AttentionPooling) CLIP models that scales well with model size and image resolution
