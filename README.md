# Awesome-ai-models

> a curated list of awesome ai models

# Environments

- Server environment
- Local Models

## Server environment:

- Localai
- Transformers

## Local environment Models:

- Transformers.js
- api
- core-ml
- windows-ml
- ml-kit

---

# Server Environments:

- # Localai

## NLP (Natural Language Processing)

### [&#x2713;] baichuan-vicuna-7b.ggmlv3.q4_0

Baichuan-7B is an open-source large-scale pre-trained model developed by Baichuan Intelligent Technology. Based on the Transformer architecture, it is a model with 7 billion parameters trained on approximately 1.2 trillion tokens. It supports both Chinese and English, with a context window length of 4096.

#### STEPS:

Download the model-> wget https://huggingface.co/TheBloke/baichuan-vicuna-7B-GGML/resolve/main/baichuan-vicuna-7b.ggmlv3.q4_0.bin -O models/baichuan-vicuna-7b.ggmlv3.q4_0

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "baichuan-vicuna-7b.ggmlv3.q4_0",
"messages": [{"role": "user", "content": "i need a coffee for"}],
"temperature": 0.9
}'

- [&#x2713;] rest api

### [&#x2713;] ggml-gpt4all-j

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

- [&#x2713;] rest api

### [] ggml-mpt-7b-base

MPT-7B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. This model was trained by MosaicML.

- this model generates number of possible answers for a given question.

#### STEPS:

Download the model-> wget https://gpt4all.io/models/ggml-mpt-7b-base.bin -O models/ggml-mpt-7b-base.bin

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "ggml-mpt-7b-base",
"messages": [{"role": "user", "content": "How are you?"}],
"temperature": 0.9
}'

- [&#x2713;] rest api

### [&#x2713;] ggml-mpt-7b-instruct

This model generates the better instructions that describes a task. Write a response that appropriately completes the request.

#### STEPS:

Download the model-> wget https://gpt4all.io/models/ggml-mpt-7b-instruct.bin -O models/ggml-mpt-7b-instruct.bin

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "ggml-mpt-7b-instruct",
"messages": [{"role": "user", "content": "teach me how to use linux terminal"}],
"temperature": 0.9
}'

##### response:

- need a tutorial that will teach me the basics of using it.

- [&#x2713;] rest api

### [&#x2713;] open-llama-7B-open-instruct.ggmlv3.q4_0

Below is an instruction that describes a task. Write a response that appropriately completes the request

    Instruction: {{.Input}}
     Response: I'm new to linux and iptables. I want to know how to use them to block specific ip addresses from connecting to my server.

#### STEPS:

    Download the model-> wget https://huggingface.co/TheBloke/open-llama-7b-open-instruct-GGML/resolve/main/open-llama-7B-open-instruct.ggmlv3.q4_0.bin -O models/open-llama-7B-open-instruct.ggmlv3.q4_0

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "open-llama-7B-open-instruct.ggmlv3.q4_0",
"messages": [{"role": "user", "content": "teach me how to use linux terminal"}],
"temperature": 0.9
}'

-[&#x2713;] rest api

### [&#x2713;] open-llama-7b-q4_0

    This model fills the sentence with the most appropriate word.

#### STEPS:

Download the model-> wget https://huggingface.co/SlyEcho/open_llama_7b_ggml/resolve/main/open-llama-7b-q4_0.bin -O models/open-llama-7b-q4_0

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "open-llama-7b-q4_0",
"messages": [{"role": "user", "content": "i had a dream to control my"}],
"temperature": 0.9
}'

- [&#x2713;] rest api

### [&#x2713;] open-llama-3b-q4_0

    This model fills the sentence with the most appropriate word.

#### STEPS:

Download the model-> wget https://huggingface.co/SlyEcho/open_llama_3b_ggml/resolve/main/open-llama-3b-q4_0.bin -O models/open-llama-3b-q4_0

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "open-llama-3b-q4_0",
"messages": [{"role": "user", "content": "i had a dream to control my"}],
"temperature": 0.9
}'

- [&#x2713;] rest api

# Local Environments:

## api

### Steps to call the api:

url = `https://api-inference.huggingface.co/models/${model_name}` (ex. model_name = stabilityai/stable-diffusion-2-1)

- curl https://api-inference.huggingface.co/models/bert-base-uncased -H "Content-Type: application.json" -d '{"inputs":"make a gun image for [MASK] ",""responseType":"blob" }'

I mentioned the models below for each experience with related few working models sir.

## fill-mask:

    related models:
    - bert-base-uncased
    - roberta-base
    - roberta-large

    - NOTE: THis model needs [MASK] token for the input sentence.

ex-request:

- curl https://api-inference.huggingface.co/models/bert-base-uncased -H "Content-Type: application.json" -d '{"inputs":"make a gun image for [MASK] ",""responseType":"blob" }'

## Question Answering:

    related models:
    - alphakavi22772023/bertQA
    - timpal0l/mdeberta-v3-base-squad2
    - LLukas22/all-MiniLM-L12-v2-qa-en

## Sentence Similarity:

    related models:
    - intfloat/e5-large-v2
    - jinaai/jina-embedding-s-en-v1
    - LLukas22/all-MiniLM-L12-v2-qa-en

## Summarization:

    related models:
    - facebook/bart-large-cnn
    - csebuetnlp/mT5_multilingual_XLSum
    - slauw87/bart_summarisation

## Text Classification:

    related models:
    - SamLowe/roberta-base-go_emotions
    - ProsusAI/finbert
    - nlptown/bert-base-multilingual-uncased-sentiment

## Text Generation:

    related models:
    - gpt2
    - HuggingFaceH4/starchat-beta-
    - fxmarty/tiny-testing-gpt2-remote-code

## Text-to-text:

        related models:
        - humarin/chatgpt_paraphraser_on_T5_base
        - juierror/flan-t5-text2sql-with-schema
        - voidful/context-only-question-generator

## Token Classification:

    related models:
    - dslim/bert-base-NER
    - DOOGLAK/01_SR_500v9_NER_Model_3Epochs_AUGMENTED
    - DOOGLAK/Paraphrased_500v3_NER_Model_3Epochs_AUGMENTED

## Translation:

    related models:
    - Helsinki-NLP/opus-mt-fi-mtt5-base
    - Helsinki-NLP/opus-mt-fr-tvl
    - Helsinki-NLP/opus-mt-eu-ru

## Zero-Shot Classification:

    related models:
    - MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
    - Narsil/deberta-large-mnli-zero-cls
    - HiTZ/A2T_RoBERTa_SMFA_TACRED-re

# Computer Vision

## Image Classification:

    related models:
    - google/vit-base-patch16-224
    - microsoft/swinv2-tiny-patch4-window8-256
    - victor/autotrain-satellite-image-classification-40975105875

## Image Segmentation:

    related models:
    - mattmdjaga/segformer_b2_clothes
    - nvidia/segformer-b3-finetuned-ade-512-512
    - zoheb/mit-b5-finetuned-sidewalk-semantic

## Object Detection:

    related models:
    - facebook/detr-resnet-50
    - hustvl/yolos-tiny\
    - facebook/detr-resnet-50-dc5
    - facebook/detr-resnet-101

# Audio

## Automatic Speech Recognition:

    related models:
    - openai/whisper-tiny.en
    - m3hrdadfi/wav2vec2-large-xlsr-persian-v3
    - facebook/s2t-medium-librispeech-asr

# Multi-modal

## Feature Extraction:

    related models:
    - facebook/bart-large
    - cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    - DeepPavlov/rubert-base-cased-sentence

## Image-to-Text:

    related models:
    - nlpconnect/vit-gpt2-image-captioning
    - microsoft/trocr-small-printed
    - naver-clova-ix/donut-base-finetuned-cord-v1

## Zero-Shot Image Classification:

    related models:
    -  openai/clip-vit-large-patch14\
    - laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    - laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K

## Text-to-image:

    related models:
    - stabilityai/stable-diffusion-2-1

    ex-request:
    - curl https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1 -H "Content-Type: application.json" -d '{"inputs":"make a  gun image  ",""responseType":"blob" }' --output output.jpg

    [X] TransformersJS WEB
    [&#x2713;] Proxy
    

- [NOTE] ( --output output.jpg) is used to save the output image.
