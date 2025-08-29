<!-- antes de enviar a versão final, solicitamos que todos os comentários, colocados para orientação ao aluno, sejam removidos do arquivo -->
# Monitoring the Impact of Offshore Oil and Gas Extraction on Calcareous Algae Environmental through Noisy Dataset

#### Aluno: [Vitor Bento de Sousa](https://github.com/vitorbds)
#### Orientadora: [Manoela Rabello Kohler](https://github.com/manoelakohler).


---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".


---

### Resumo
Conjuntos de dados do mundo real frequentemente contêm amostras ruidosas, ou seja, instâncias rotuladas incorretamente, o que pode impactar significativamente a robustez e a capacidade de generalização dos modelos de Deep Learning. Para enfrentar esse desafio, métodos de ponta frequentemente adotam a estratégia de small-loss, que parte da premissa de que amostras rotuladas corretamente tendem a apresentar perdas de treinamento mais baixas. Com base nessa abordagem, o framework Retrieving Discarded Samples (RDS) foi recentemente introduzido como um mecanismo para recuperar amostras potencialmente informativas inicialmente excluídas durante o treinamento.

Neste estudo, demonstramos que o desempenho do framework RDS é altamente sensível à qualidade dos pseudo-rótulos atribuídos a essas amostras recuperadas. Para melhorar esse aspecto, introduzimos o RDS-Contrastive Learning, uma nova variante que incorpora aprendizado auto-supervisionado para aprimorar a precisão dos pseudo-rótulos e o desempenho geral do modelo.

Avaliamos nosso modelo em quatro diferentes conjuntos de dados de benchmark, incluindo o dataset Clothing1M, alcançando até 4% de melhoria no F1-score em relação a métodos de ponta existentes. Nosso modelo foi aplicado à conservação ambiental. Ele foi utilizado para classificar algas calcárias e monitorar os impactos da extração de petróleo e gás offshore no ambiente marinho. Dada a natureza das algas calcárias, o conjunto de dados de treinamento disponível era ruidoso, e nosso modelo obteve uma melhoria de 2,5% no F1-score em comparação a outros modelos de ponta.

### Abstract <!-- Opcional! Caso não aplicável, remover esta seção -->

Real-world datasets frequently contain noisy samples, i.e. mislabeled instances, which can significantly impact the robustness and generalization of Deep Learning models. To address this challenge, state-of-the-art methods often adopt the small-loss strategy, which assumes that correctly labeled samples tend to yield lower training losses. Building upon this approach, the Retrieving Discarded Samples (RDS) framework was recently introduced as a mechanism to recover potentially informative samples initially excluded during training. In this study, we demonstrate that the performance of the RDS framework is highly sensitive to the quality of pseudo-labels assigned to these retrieved samples. To improve this aspect, we introduce RDS-Contrastive Learning, a novel variant that incorporates self-supervised learning to enhance pseudo-label accuracy and overall model performance. We evaluate our model on four different benchmark datasets, including the Clothing1M dataset, achieving up to a 4\% improvement in F1-score over existing state-of-the-art methods. Our model was applied to environmental conservation. It was used to classify calcareous algae and monitor the impacts of offshore oil and gas extraction on the marine environment. Given the nature of calcareous algae, the available training dataset was noisy, and our model achieved a 2.5\% improvement in F1-score compared to other state-of-the-art models.
### 1. Introdução

Um desafio comum em conjuntos de dados reais é a presença de amostras ruidosas, ou seja, instâncias rotuladas incorretamente, que prejudicam o desempenho e a capacidade de generalização de modelos de Deep Learning. Em um estudo recente da indústria de óleo e gás, foram avaliados modelos de ponta para lidar com ruído nos rótulos em uma tarefa de monitoramento ambiental voltada à classificação de algas calcárias. O modelo RDS-C obteve o melhor desempenho.

O ruído nos rótulos desse conjunto de dados surgiu, principalmente, de anotações feitas por não especialistas, anotadores fatigados e erros de rotulagem automática ou via crowdsourcing. Até mesmo especialistas podem cometer erros diante de classificações difíceis.

A estratégia Small Loss Approach (SLA) é amplamente usada para esse problema, assumindo que amostras corretas geram menores perdas de treinamento, enquanto amostras de alta perda são descartadas por serem potencialmente ruidosas. Para aprimorar esse método, foi proposto o framework Retrieving Discarded Samples (RDS), que tenta recuperar as amostras descartadas atribuindo pseudo-rótulos a elas. O modelo RDS-C combina essa abordagem com o paradigma de Co-teaching, em que duas redes trocam informações para aumentar a robustez contra ruído.

No entanto, o desempenho do RDS depende da identificação correta das amostras ruidosas e da qualidade dos pseudo-rótulos, o que nem sempre ocorre na prática. Para enfrentar essa limitação, o estudo integrou aprendizado auto-supervisionado e contrastivo ao RDS-C, resultando no RDS-Contrastive. Essa versão adiciona uma nova função de perda baseada em aprendizado contrastivo para melhorar o reconhecimento de padrões entre redes gêmeas e a precisão dos pseudo-rótulos, além de ponderar a confiança atribuída a eles.

O modelo foi avaliado em quatro benchmarks (CIFAR-10, CIFAR-100, MNIST e Clothing1M) e em um conjunto real da indústria de óleo e gás para classificação de algas calcárias, caracterizado por alta similaridade entre classes e incerteza nos rótulos. Nos testes reais, o RDS-Contrastive obteve um ganho de 2,4 pontos percentuais no F1-score em relação aos modelos de ponta, mostrando sua aplicabilidade prática no monitoramento ambiental marinho frente aos impactos da extração offshore.

### 2. Modelagem

A precisão dos pseudo-rótulos atribuídos durante o processo de RDS-Label impacta fortemente o desempenho do modelo. Modelos de deep learning treinados com menos amostras ruidosas tendem a generalizar melhor, e, por isso, aprimorar o mecanismo de pseudo-rotulagem é uma forma promissora de aumentar a robustez.

Outra motivação é que treinar apenas com rótulos ruidosos equivale, na prática, a treinar sem rótulos, tornando o aprendizado auto-supervisionado essencial. Com base nisso, o modelo proposto integra técnicas de aprendizado contrastivo ao framework RDS.

O processo começa aplicando o Small Loss Approach (SLA) para separar amostras limpas das ruidosas. As amostras limpas são usadas diretamente no treinamento, enquanto as ruidosas passam pelo mecanismo de pseudo-rotulagem. Em paralelo, todas as amostras também são processadas por um módulo de aprendizado contrastivo que reforça a aprendizagem de representações.

O objetivo de treinamento combina três componentes:

perda supervisionada sobre amostras limpas,

perda com pseudo-rótulos sobre amostras ruidosas,

perda auto-supervisionada contrastiva.

Nesse processo, duas transformações de aumento de dados são aplicadas a cada amostra, gerando duas visões correlacionadas. A perda contrastiva é então calculada, incentivando o modelo a aproximar representações de instâncias semelhantes e afastar as de instâncias diferentes. Esse termo atua como regularizador e aumenta a robustez do modelo diante de ruído nos rótulos.



### 3. Resultados
Nos experimentos realizados, o modelo RDS-Contrastive apresentou ganhos consistentes em diferentes cenários. Nos quatro conjuntos de dados de benchmark avaliados — CIFAR-10, CIFAR-100, MNIST e Clothing1M — obteve até 4% de melhoria no F1-score em relação aos métodos de ponta existentes. Já na aplicação prática em conservação ambiental, voltada à classificação de algas calcárias para monitorar os impactos da extração offshore de petróleo e gás, o modelo alcançou um ganho adicional de 2,4 pontos percentuais no F1-score quando comparado aos melhores modelos de referência. Em outro conjunto específico da mesma aplicação, caracterizado por alto ruído nos rótulos, foi observada uma melhoria de 2,5% no F1-score, evidenciando a capacidade do RDS-Contrastive de lidar eficazmente com dados ruidosos em cenários reais e complexos.

### 4. Conclusões

O RDS-Contrastive também demonstrou resultados fortes e consistentes em conjuntos de dados de benchmark, incluindo CIFAR-10, CIFAR-100, MNIST e o dataset de larga escala Clothing1M — mesmo não tendo sido projetado especificamente para cenários de ruído em open-set. As melhorias de desempenho chegaram a 2,5% em acurácia de teste e 3,0% em F1-score em comparação com métodos concorrentes.

Além disso, uma análise aprofundada do mecanismo de RDS-Labeling mostrou que o modelo proposto mantém um equilíbrio favorável entre a precisão dos pseudo-rótulos e o desempenho final de classificação. Esses resultados destacam a capacidade de generalização do modelo e o impacto positivo da integração do aprendizado contrastivo ao framework RDS.

O aprendizado com rótulos ruidosos continua sendo uma área desafiadora, relevante e emergente de pesquisa, especialmente em cenários de múltiplas classes e aplicações reais. Estudos futuros devem ir além dos benchmarks tradicionais e priorizar conjuntos de dados específicos de domínio, a fim de capturar melhor as nuances e restrições dos dados do mundo real.

Também é recomendada a exploração de adaptações do RDS-Contrastive para cenários de open-set ou domain-shift. Além disso, o desenvolvimento de modelos menos sensíveis à escolha de hiperparâmetros aumentaria sua acessibilidade e facilidade de uso em ambientes de produção. Reduzir a dependência de calibrações específicas de domínio pode ainda ampliar a aplicabilidade de técnicas de aprendizado robusto a ruído em diferentes áreas.

Os ganhos consistentes do RDS-Contrastive sobre o modelo RDS-C reforçam ainda mais a importância do termo contrastivo. Como trabalho futuro, uma direção promissora seria estender essa ideia incorporando o termo contrastivo ao modelo RDS-J, o que pode levar a melhorias adicionais.m.

---

Matrícula: 192.190.004

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
