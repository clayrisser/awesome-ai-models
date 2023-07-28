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

### [&#x2713;] ggml-mpt-7b-base

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

### [&#x2713;] open-llama-3b-q4_0

    This model fills the sentence with the most appropriate word.

#### STEPS:

Download the model-> wget https://huggingface.co/SlyEcho/open_llama_7b_ggml/resolve/main/open-llama-7b-q4_0.bin -O models/open-llama-7b-q4_0

##### request:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "open-llama-3b-q4_0",
"messages": [{"role": "user", "content": "i had a dream to control my"}],
"temperature": 0.9
}'

- [&#x2713;] rest api
