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

### Experience:

#### Models: []

#### Description:

## Computer Vision

### Model name

#### Experience:

#### Description:
