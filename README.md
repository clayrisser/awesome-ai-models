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

# Localai

## NLP (Natural Language Processing)

### [&#x2713;] baichuan-vicuna-7b.ggmlv3.q4_0

Baichuan-7B is an open-source large-scale pre-trained model developed by Baichuan Intelligent Technology. Based on the Transformer architecture, it is a model with 7 billion parameters trained on approximately 1.2 trillion tokens. It supports both Chinese and English, with a context window length of 4096.

#### STEPS:

Download the model-> WGET https://huggingface.co/TheBloke/baichuan-vicuna-7B-GGML/resolve/main/baichuan-vicuna-7b.ggmlv3.q4_0.bin -O models/baichuan-vicuna-7b.ggmlv3.q4_0

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
