# Objetivo

*Churn* pode ser entendido como o ato, feito por parte do cliente, de cancelar um determinado serviço. Também conhecido como atrito, esse é um grande problema enfrentado por empresas que dependam da fidelidade, como as que oferecem *Software as a Service* ou serviços financeiros como Cartões de Crédito.

A base de dados apresentada aqui trata justamente do problema do *Churn* em uma empresa de cartão de crédito. O interesse aqui é tentar entender os principais fatores por trás da decisão de cancelar o cartão, e posteriormente tentar prever quais clientes entrarão em atríto. Sabendo disso é possível criar políticas focadas nos clientes com potencial de atrito, aumentando a taxa de fidelidade com a empresa.

# Dataset

Os dados são retirados do [Kaggle](https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn) e são compostos de 21 variáveis, incluindo atributos sobre o cliente, sobre a relação do cliente com a provedora do cartão e sobre como eles utilizam o cartão.

# Tecnologias

A aplicação foi toda feita em Python, utilizando o Poetry para gerenciar as dependências e o GNU Make para gerenciar toda a pipeline de leitura, processamento dos dados, treinamento e avaliação dos modelos.

A leitura dos dados foi feita com o Pandas e o processamento dos dados para modelagem foram feitas com o numpy, sklearn e o pacote category encoders. Visualizações foram feitas com o Altair. Apliquei quatro modelos nos dados: Regressão Logística, Árvores de Decisão (Decision Trees) e Florestas Aleatórias (Random Forests). Todos os modelos e otimização dos parâmetros foram feitos com o sklearn.

# Como Rodar o Projeto

O projeto precisa ter o Python instalado, além do [Poetry](https://python-poetry.org/docs/master/) e do [GNU Make](https://www.gnu.org/software/make/).

Com todos instalados, clone este repositório e rode o make com o seguinete comando:

``` bash
make all
```

Ao final ele dará a performance do melhor modelo no conjunto de teste entre todos os avaliados.

Opcionalmente, é possível definir alguns parâmetros para o treinamento dos modelos:

* TEST_SIZE: A proporção da base de dados que é divida para o conjunto de teste. Deve ser uma float entre 0 e 1. O padrão é 0.2;
* RANDOM_STATE: Um número inteiro definindo o random state para replicação dos dados. O padrão é 42.
* CV: O número de folds de cross-validation utilizado em RandomSearchCV. O padrão é 5.
* N_ITER: O número de iterações utilizadas por RandomSearchCV para testar diferentes hiperparâmetros. O padrão 50.
* SCORING: A métrica do sklearn utilizada para definir o melhor modelo. O padrão é roc_auc
* THRESHOLD: A probabilidade limite a partir do qual atribui uma predição para a classe positiva. O padrão é 0.22.

Que podem ser usados da seguinte forma, modificando um ou mais parâmetros:

``` bash
make all TEST_SIZE=0.2 CV=20 ...
```

# Análises

* [Exploração dos Dados](notebooks/1.0-data-exploration.ipynb)
* [Modelagem](notebooks/2.0-modeling.ipynb)
* [Análise dos Modelos](notebooks/3.0-model-analysis.ipynb)
* Post no Blog (Em construção)

# Contato

* [LinkedIn](https://www.linkedin.com/in/silasge/)
* [Twitter](https://twitter.com/_silasge)
* [sg.lopes26@gmail.com](mailto:sg.lopes26@gmail.com)