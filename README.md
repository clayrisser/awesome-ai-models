# awesome-ai-models

> a curated list of awesome ai models

# TOPOLOGY: Transformers

## NLP (Natural Language Processing)

## baichuan-vicuna-7b.ggmlv3.q4_0

Baichuan-7B is an open-source large-scale pre-trained model developed by Baichuan Intelligent Technology. Based on the Transformer architecture, it is a model with 7 billion parameters trained on approximately 1.2 trillion tokens. It supports both Chinese and English, with a context window length of 4096.

### STEPS:

Download the model-> WGET https://huggingface.co/TheBloke/baichuan-vicuna-7B-GGML/resolve/main/baichuan-vicuna-7b.ggmlv3.q4_0.bin -O models/baichuan-vicuna-7b.ggmlv3.q4_0

#### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "baichuan-vicuna-7b.ggmlv3.q4_0",
"messages": [{"role": "user", "content": "i need a coffee for"}],
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

### Models:

#### Description:

## Computer Vision

### Model name

#### Experience:

#### Description:
