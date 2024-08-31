# Context

- LLMによる回答生成能力向上を狙い、Mixture of Agents（MoA）を実装する。
  - [Together MoA — collective intelligence of open-source models pushing the frontier of LLM capabilities](https://www.together.ai/blog/together-moa)
  - MoAは複数のLLMによる回答生成を階層化し、各階層での回答を集約していく。各階層では、3つのLLMが同じ問に対して回答を生成する。生成した回答は合体され、一つのテキストとなったのち、次階層に送られ、次階層にある複数LLMへのインプットとなる。
  - すべての階層を抜けた後、1つのLLMによって、階層を経た回答を使って、最終的な回答を生成する。
  - この最終回答の生成には、パラメタ数の多いLLMを使うことでより精度が高くなる。
  - しかしながら、パラメータ数の少ないLLMを使う場合においても、MoAを実装することで、実装しない場合に比べて回答精度が情報することが報告されていル。
    - これは、パラメタ数の少ないLLMしか利用できない環境であっても、回答精度を上昇させられることを意味している。

# Coding
## Overview
- name: `mixture_of_agent_request.py`
- description:
  - このコードでは、OpenRouter または groq を利用して、複数のLLMを使用する
  - LLMのIFには LangChain を利用する

## Flow
1. 

