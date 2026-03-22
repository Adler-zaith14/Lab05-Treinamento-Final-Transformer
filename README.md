# Lab 05 — Treinamento Fim-a-Fim do Transformer

**Instituição:** ICEV — Instituto de Ensino Superior  
**Disciplina:** Tópicos em Inteligência Artificial  
**Professor:** Dimmy Magalhães  
**Autor:** Adler Castro Alves  

---

## O que é esse laboratório

Esse é o laboratório final da Unidade I. O objetivo foi pegar o modelo Transformer que construí nos labs anteriores e finalmente treiná-lo de verdade, conectando a um dataset real e escrevendo o loop de treinamento completo.

A ideia não era fazer um tradutor perfeito — isso precisaria de dias de treino em GPUs caras. O que precisava provar era que a arquitetura consegue aprender, ou seja, que o loss cai ao longo das épocas.

---

## Como rodar

O projeto foi desenvolvido no Google Colab. Para rodar, basta abrir o arquivo `lab-5_la.py` no Colab e executar as células em ordem. As dependências são instaladas na primeira célula:

```
pip install datasets transformers torch
```

---

## O que foi feito em cada tarefa

**Tarefa 1 — Dataset**  
Carreguei o dataset `bentrevett/multi30k` do Hugging Face, que tem pares de frases em inglês e alemão. Usei só as primeiras 1000 frases para o treino rodar rápido no Colab gratuito.

**Tarefa 2 — Tokenização**  
Usei o tokenizador `bert-base-multilingual-cased` para converter as frases em listas de números. No lado do Decoder coloquei o token `[CLS]` como início da frase e `[SEP]` como fim, e apliquei padding pra todos os batches terem o mesmo tamanho (32 tokens).

**Tarefa 3 — Training Loop**  
Instanciei o modelo com dimensões pequenas pra caber no Colab (`d_model=128`, 4 cabeças, 2 camadas). Usei `CrossEntropyLoss` com `ignore_index` no padding e o otimizador Adam. Treinei por 15 épocas e o loss caiu de 8.75 até 0.81, o que mostra que o modelo estava aprendendo de verdade.

**Tarefa 4 — Overfitting Test**  
Peguei a primeira frase do conjunto de treino e rodei o loop auto-regressivo pra ver se o modelo conseguia reproduzir a tradução. O resultado foi bem próximo do esperado:

- Entrada (EN): *Two young, white males are outside near many bushes.*
- Esperado (DE): *Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.*
- Gerado: *Zwei junge weiße Männer sind im Freien in der Nähe des Gebäudeten vorführen.*

O começo da frase saiu quase idêntico, o que confirma que os gradientes estão fluindo certo.

---

## Arquitetura implementada

Toda a arquitetura foi construída do zero em PyTorch, sem usar o `nn.Transformer` pronto da biblioteca.

O ponto de partida é a função `atencao()`, que implementa a atenção escalada — ela recebe as matrizes de query, key e value, calcula os scores dividindo pela raiz de dk e aplica softmax para obter os pesos. Em cima disso foi construída a classe `MultiHeadAtencao`, que divide a atenção em múltiplas cabeças paralelas usando as projeções WQ, WK, WV e WO, permitindo que o modelo preste atenção em diferentes partes da sequência ao mesmo tempo.

Para as outras sub-camadas, foi implementada a `FFN` (rede feed-forward com ReLU entre duas camadas lineares) e a `AddNorm`, que combina a conexão residual com a Layer Normalization. A codificação posicional ficou na classe `PositionalEncoding`, que usa funções de seno e cosseno para injetar informação de posição nos embeddings. No Decoder, a função `mascara_causal()` gera a máscara triangular inferior que impede o modelo de enxergar tokens futuros durante o treinamento.

Com esses blocos, foram montadas as classes `BlocoEncoder` e `BlocoDecoder`. O bloco do Encoder empilha a atenção com as sub-camadas de normalização e feed-forward, enquanto o bloco do Decoder adiciona uma segunda camada de atenção cruzada (cross-attention) que conecta a saída do Encoder com o que está sendo gerado. O `Encoder` e o `Decoder` finais são apenas N repetições desses blocos, com o Decoder tendo ainda uma camada de projeção linear no final para mapear para o vocabulário.

---

**Anexo Google Colab:**
[https://colab.research.google.com/drive/1V36WJqt8wtM32eSqJ2tNa2z9Z0GuLlEx?usp=sharing]
**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..

---

## Versionamento

```
git tag v1.0
```
