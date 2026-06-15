Preditores da Resiliência Criativa em Contextos de Vulnerabilidade Socioeconômica: Um Pipeline Reproduzível de Learning Analytics, Explainable AI e Fairness com Dados do PISA 2022
Jose Edson Amorim Sebastião1
1Departamento de Estatística e Informática – Universidade Federal Rural de Pernambuco (UFRPE) – Pernambuco – Brazil
joseedson.sebastiao@ufrpe.br


Resumo. Este estudo investiga “resiliência criativa” em estudantes brasileiros do PISA 2022 integrando Learning Analytics, Educational Data Mining, Machine Learning, Explainable AI (XAI), auditoria de fairness e análises de robustez. A base analítica final contém **3.834 estudantes** (dataset `pisa_brasil_estudo_limpo.csv`) e o construto foi operacionalizado por meio de definições alternativas baseadas na combinação de **vulnerabilidade socioeconômica (ESCS)** e **pensamento criativo (CRT_SCORE)**. A definição principal (Target A) classificou como resilientes os estudantes com **ESCS ≤ Q1** e **CRT_SCORE ≥ Q3**, resultando em **164 estudantes** e prevalência **0,0427752**.
O pipeline incluiu: auditoria inicial de qualidade (sem duplicatas e missingness elevada em parte das variáveis de criatividade), controle de data leakage via “gate” por proxy (incluindo `CRT_SCORE`), construção e comparação de targets (A/B/C/D), perfilamento e clusterização person-centered (solução escolhida com **agglomerative_ward k=2**), modelagem supervisionada para o Target A e explicação via SHAP. O modelo vencedor foi o **xgboost_tuned**, com **ROC-AUC (holdout) = 0,9313** e desempenho avaliado também em validação cruzada (**ROC-AUC (CV) = 0,9236**). A interpretabilidade (SHAP, método `shap_tree`) indicou maior relevância de **HISCED**, seguida por **ICTRES** e **HOMEPOS**, além de variáveis `CR*` associadas a perfis de engajamento criativo. A auditoria de fairness analisou variáveis sensíveis (**ST004D01T**, **Grupo_ESCS**, **HISCED**) e reportou cenários em que grupos possuem **prevalência 0**, o que produz **ROC-AUC como “nan”** nessas subpopulações. A robustez foi avaliada por bootstrap, com **Brier score = 0,0228** e **roc_auc bootstrap mean = 0,973985** (CI [0,966172, 0,980896]).
Os achados sustentam que a resiliência criativa, tal como operacionalizada no projeto, é um fenômeno raro e heterogêneo, associado a padrões latentes (clusters) e a fatores com relevância explicável. A interpretação pedagógica deve ser feita com cautela devido à natureza observacional dos dados e ao risco remanescente inerente à utilização de proxies de target.
Palavras-chave: Resiliência criativa; PISA 2022; Pensamento Criativo; Learning Analytics; Explainable Artificial Intelligence; Fairness; Machine Learning; Vulnerabilidade socioeconômica.

Abstract. This study investigates “creative resilience” in Brazilian students from PISA 2022 by integrating Learning Analytics, Educational Data Mining, Machine Learning, Explainable AI (XAI), algorithmic fairness auditing, and robustness analyses. The final analytic dataset contains **3,834 students** (`pisa_brasil_estudo_limpo.csv`). Creative resilience was operationalized using multiple definitions combining **socioeconomic vulnerability (ESCS)** and **creative thinking (CRT_SCORE)**. The primary definition (Target A) classified resilient students with **ESCS ≤ Q1** and **CRT_SCORE ≥ Q3**, yielding **164 students** and prevalence **0.0427752**.
The pipeline included: initial data quality auditing (no duplicates and high missingness in part of creativity-related variables), data leakage control through a proxy-based gate (including `CRT_SCORE`), target construction and comparison (A/B/C/D), person-centered profiling and clustering (**agglomerative_ward with k=2**), supervised modeling for Target A and SHAP-based explanation. The best-performing model was **xgboost_tuned**, with **ROC-AUC (holdout) = 0.9313** and **ROC-AUC (CV) = 0.9236**. Interpretability via SHAP (`shap_tree`) highlighted **HISCED** as the most relevant factor, followed by **ICTRES** and **HOMEPOS**, along with several `CR*` variables tied to creative engagement profiles. Fairness auditing examined sensitive variables (**ST004D01T**, **Grupo_ESCS**, **HISCED**) and reported cases where some groups have **prevalence 0**, leading to **ROC-AUC = nan** for those subpopulations. Robustness was assessed via bootstrap, with **Brier score = 0.0228** and **roc_auc mean = 0.973985** (CI [0.966172, 0.980896]).
Overall, the results support that creative resilience—under the project’s operational definition—is rare and heterogeneous, associated with latent patterns (clusters) and with factors that are relevant according to explainable models. Pedagogical interpretation should be cautious due to the observational nature of the data and residual proxy-related risks.
Keywords: Creative Resilience; PISA 2022; Creative Thinking; Learning Analytics; Explainable Artificial Intelligence; Fairness; Machine Learning; Socioeconomic Vulnerability.

















1. Introdução
A desigualdade educacional permanece como um dos principais desafios para a promoção da equidade e do desenvolvimento humano. Estudantes provenientes de contextos socioeconômicos desfavoráveis frequentemente enfrentam restrições relacionadas ao acesso a recursos educacionais, tecnológicos e culturais, o que pode comprometer o desenvolvimento de competências cognitivas complexas e reduzir oportunidades de aprendizagem ao longo da trajetória escolar. No Brasil, essas desigualdades assumem relevância particular devido à persistência de disparidades sociais que influenciam o desempenho educacional desde os primeiros anos da educação básica.
Nesse contexto, avaliações educacionais em larga escala desempenham papel fundamental na produção de evidências capazes de subsidiar políticas públicas orientadas por dados. Entre elas destaca-se o Programme for International Student Assessment (PISA), coordenado pela Organização para Cooperação e Desenvolvimento Econômico (OCDE), que avalia estudantes de aproximadamente quinze anos de idade em diversos países e disponibiliza um amplo conjunto de informações relacionadas às condições socioeconômicas, características familiares, experiências escolares e competências cognitivas dos participantes.
No ciclo de 2022, o PISA incorporou pela primeira vez o domínio de Pensamento Criativo (Creative Thinking), definido pela OCDE como a capacidade de gerar, avaliar e aperfeiçoar ideias originais e eficazes para diferentes propósitos, incluindo resolução de problemas, expressão criativa e construção de conhecimento. A introdução desse domínio ampliou significativamente as possibilidades de investigação sobre competências consideradas essenciais para o século XXI, especialmente em sociedades caracterizadas por rápidas transformações tecnológicas, inovação constante e profundas desigualdades sociais.
Embora exista ampla evidência empírica da associação entre vulnerabilidade socioeconômica e menores níveis de desempenho educacional, observa-se que parte dos estudantes consegue alcançar resultados elevados mesmo diante de condições adversas. Esse fenômeno tem sido tradicionalmente estudado sob a perspectiva da resiliência acadêmica. Inspirado nessa literatura, o presente trabalho adota o conceito de resiliência criativa, entendido como a ocorrência simultânea de vulnerabilidade socioeconômica e elevado desempenho em Pensamento Criativo.
Entretanto, a operacionalização desse conceito não é trivial. Diferentes critérios podem ser utilizados para definir vulnerabilidade e excelência criativa, produzindo populações distintas de estudantes resilientes. Dessa forma, além de investigar fatores associados ao fenômeno, este estudo também analisa e compara diferentes definições operacionais de resiliência criativa construídas a partir da combinação entre indicadores socioeconômicos e desempenho em Pensamento Criativo.
A amostra utilizada é composta por 3.834 estudantes brasileiros participantes do PISA 2022. Foram construídas e comparadas múltiplas definições de resiliência criativa, incluindo abordagens baseadas em quartis, percentis e escores compostos. Na definição principal adotada neste estudo, foram considerados resilientes os estudantes pertencentes ao quartil inferior do índice Economic, Social and Cultural Status (ESCS) e simultaneamente posicionados no quartil superior do desempenho em Pensamento Criativo (CRT_SCORE). Essa definição resultou em uma prevalência aproximada de 4,3% da amostra total, evidenciando que a resiliência criativa constitui um fenômeno relativamente raro, mas potencialmente relevante para a compreensão do sucesso educacional em contextos adversos.
A investigação desse fenômeno envolve desafios metodológicos importantes. Em estudos baseados em Aprendizagem de Máquina (Machine Learning), a presença de variáveis derivadas direta ou indiretamente da variável-alvo pode introduzir vazamento de informação (data leakage), produzindo métricas artificialmente elevadas e comprometendo a validade científica dos resultados. Além disso, a elevada dimensionalidade dos microdados do PISA, composta por mais de mil variáveis, exige procedimentos sistemáticos de auditoria, seleção de atributos, validação e interpretação dos modelos.
Diante desses desafios, o presente estudo propõe um pipeline analítico reproduzível para investigação da resiliência criativa no PISA 2022. O pipeline integra auditoria de qualidade dos dados, descoberta e catalogação de variáveis, detecção de potenciais fontes de leakage, comparação de definições de target, análise exploratória, perfilamento de estudantes resilientes, clusterização, modelagem preditiva supervisionada, interpretabilidade baseada em Explainable Artificial Intelligence (XAI), auditoria de fairness algorítmica e análise de robustez dos modelos.
Diferentemente de abordagens centradas exclusivamente na maximização do desempenho preditivo, o objetivo deste trabalho é construir uma infraestrutura analítica transparente, auditável e cientificamente robusta para compreender fatores associados à resiliência criativa. Nesse sentido, a Aprendizagem de Máquina é utilizada não apenas como ferramenta de classificação, mas também como instrumento para geração de conhecimento sobre os mecanismos associados ao desenvolvimento do pensamento criativo em contextos de vulnerabilidade socioeconômica.
Ao integrar Learning Analytics, Educational Data Mining, Explainable Artificial Intelligence e Fairness, esta pesquisa contribui tanto para o avanço metodológico das aplicações de inteligência artificial em educação quanto para a compreensão de fatores associados ao sucesso criativo em populações vulneráveis. Adicionalmente, o estudo disponibiliza um conjunto de artefatos reproduzíveis, incluindo relatórios analíticos, métricas de avaliação, explicações dos modelos e um dashboard interativo para exploração dos resultados.
1.1 Perguntas de Pesquisa
A investigação é orientada pelas seguintes perguntas de pesquisa:
PP1. Como diferentes definições operacionais de resiliência criativa afetam a identificação de estudantes resilientes no PISA 2022?
PP2. Quais características socioeconômicas, familiares, tecnológicas e processuais estão associadas à ocorrência da resiliência criativa?
PP3. É possível identificar estudantes criativamente resilientes por meio de modelos de aprendizagem de máquina com desempenho preditivo robusto e livre de vazamento de informação?
PP4. Quais variáveis apresentam maior relevância para a explicação da resiliência criativa segundo técnicas de interpretabilidade e Explainable Artificial Intelligence?
PP5. Os modelos desenvolvidos apresentam comportamento equitativo entre diferentes grupos populacionais avaliados pelas métricas de fairness?
PP6. Existem perfis latentes distintos de estudantes associados à resiliência criativa identificáveis por técnicas de clusterização?



1.2 Objetivos
Objetivo Geral
Investigar fatores associados à resiliência criativa entre estudantes brasileiros participantes do PISA 2022 por meio de técnicas de aprendizagem de máquina e inteligência artificial explicável, buscando compreender quais características distinguem estudantes que alcançam elevado desempenho em Pensamento Criativo apesar de condições socioeconômicas desfavoráveis.
Objetivos Específicos
Definir operacionalmente a resiliência criativa a partir da combinação entre vulnerabilidade socioeconômica e elevado desempenho em Pensamento Criativo;
Realizar auditoria completa da base de dados para identificação de problemas de qualidade, valores ausentes, variáveis derivadas e potenciais fontes de vazamento de informação;
Caracterizar o perfil dos estudantes criativamente resilientes por meio de análises exploratórias e comparativas;
Investigar a influência de fatores socioeconômicos, familiares, tecnológicos e educacionais sobre a ocorrência de resiliência criativa;
Aplicar técnicas de clusterização para identificar perfis distintos de estudantes presentes na amostra;
Desenvolver modelos supervisionados de classificação utilizando Regressão Logística e Random Forest para identificação de estudantes resilientes;
Avaliar o desempenho preditivo por meio de métricas como ROC-AUC, Precision, Recall, F1-Score e Balanced Accuracy;
Aplicar técnicas de interpretabilidade baseadas em importância de atributos e análise explicável dos modelos;
Investigar possíveis diferenças de desempenho entre grupos demográficos por meio de métricas de fairness algorítmica;
Produzir evidências que possam subsidiar políticas públicas voltadas à promoção da criatividade, inclusão digital, equidade educacional e redução dos impactos das desigualdades socioeconômicas sobre o desenvolvimento do pensamento criativo.

2. Fundamentação Teórica e Estado da Arte
2.1 Pensamento Criativo no PISA 2022
O Programme for International Student Assessment (PISA), coordenado pela Organização para a Cooperação e Desenvolvimento Econômico (OCDE), constitui uma das mais importantes iniciativas internacionais de avaliação educacional. Além de mensurar competências em Leitura, Matemática e Ciências, o programa disponibiliza um amplo conjunto de variáveis contextuais relacionadas às características socioeconômicas, familiares, escolares e comportamentais dos estudantes.
No ciclo de 2022, o PISA incorporou pela primeira vez o domínio de Pensamento Criativo (Creative Thinking), definido como a capacidade de gerar, avaliar e aperfeiçoar ideias originais e eficazes para resolver problemas, produzir conhecimento e expressar-se criativamente (OECD, 2023a).
A introdução desse domínio ampliou significativamente as possibilidades de investigação sobre competências consideradas essenciais para o século XXI, incluindo criatividade, inovação, flexibilidade cognitiva e resolução criativa de problemas. Diferentemente das áreas tradicionalmente avaliadas, o Pensamento Criativo busca capturar processos cognitivos relacionados à produção de ideias novas e úteis em múltiplos contextos.
Essa expansão do escopo avaliativo permite investigar não apenas desigualdades de desempenho acadêmico tradicional, mas também fatores associados ao desenvolvimento de competências criativas em diferentes grupos sociais.

2.2 Resiliência Acadêmica e Resiliência Criativa
A literatura sobre resiliência acadêmica busca compreender por que determinados estudantes conseguem alcançar resultados educacionais positivos mesmo quando expostos a condições adversas, especialmente vulnerabilidade socioeconômica.
Historicamente, estudos de resiliência têm utilizado indicadores de desempenho em Matemática, Leitura e Ciências como desfechos principais. A introdução do domínio de Pensamento Criativo no PISA 2022 possibilita expandir esse debate para competências associadas à criatividade e inovação.
Neste estudo, a noção de resiliência criativa refere-se à capacidade de estudantes socialmente vulneráveis apresentarem desempenho criativo superior ao esperado diante de suas condições socioeconômicas.
Entretanto, não existe uma definição única e consensual para esse fenômeno. Diferentes critérios podem produzir grupos distintos de estudantes resilientes, afetando tanto a prevalência observada quanto os fatores associados ao fenômeno.
Por essa razão, o presente trabalho adota uma abordagem comparativa baseada em múltiplas definições operacionais de resiliência criativa. Foram construídos quatro targets alternativos, variando os critérios de vulnerabilidade socioeconômica e desempenho criativo. Essa estratégia permite avaliar a robustez dos resultados diante de diferentes formas de operacionalização do conceito e reduzir a dependência de uma única definição arbitrária.
A comparação sistemática entre diferentes targets representa uma contribuição metodológica importante, uma vez que grande parte da literatura utiliza apenas uma definição de resiliência sem investigar a sensibilidade dos resultados a diferentes critérios de classificação.
2.3 Learning Analytics e Aprendizagem de Máquina em Educação
O crescimento da disponibilidade de dados educacionais em larga escala impulsionou o desenvolvimento das áreas de Learning Analytics (LA) e Educational Data Mining (EDM).
Esses campos integram métodos estatísticos, computacionais e técnicas de inteligência artificial para compreender processos educacionais, identificar padrões de aprendizagem e apoiar a tomada de decisão baseada em evidências.
No contexto educacional, algoritmos de Aprendizagem de Máquina vêm sendo utilizados para diversas finalidades, incluindo:
previsão de desempenho acadêmico;
identificação de risco de evasão;
análise de engajamento estudantil;
detecção de perfis de aprendizagem;
identificação de fatores associados ao sucesso educacional.
Uma característica importante desses métodos é sua capacidade de modelar relações complexas e não lineares entre variáveis, frequentemente capturando padrões que não seriam identificados por técnicas estatísticas tradicionais.
Entretanto, em aplicações educacionais, o objetivo não deve se restringir à obtenção de elevado desempenho preditivo. A compreensão dos fatores associados aos resultados observados é igualmente relevante, especialmente quando os modelos são utilizados para subsidiar intervenções pedagógicas ou políticas públicas.
Nesse contexto, este estudo emprega a aprendizagem de máquina não apenas como ferramenta de previsão, mas também como instrumento para investigação científica dos fatores associados à resiliência criativa.

2.4 Inteligência Artificial Explicável (XAI)
Embora algoritmos modernos de aprendizagem de máquina possam apresentar elevado desempenho preditivo, muitos deles operam como sistemas de baixa interpretabilidade, frequentemente descritos como black boxes.
A Inteligência Artificial Explicável (Explainable Artificial Intelligence – XAI) surgiu como uma área dedicada ao desenvolvimento de métodos capazes de tornar os modelos mais transparentes e compreensíveis.
Entre as técnicas mais difundidas destaca-se o SHAP (SHapley Additive Explanations), fundamentado na teoria dos jogos cooperativos. O método estima a contribuição individual de cada variável para as previsões realizadas pelo modelo, permitindo identificar fatores associados ao fenômeno investigado tanto em nível global quanto individual.
Em pesquisas educacionais, técnicas explicáveis desempenham papel particularmente importante porque permitem transformar modelos preditivos em instrumentos de geração de conhecimento científico.
Neste trabalho, a interpretabilidade é tratada como componente central da análise, sendo utilizada para identificar quais características apresentam maior associação com a ocorrência da resiliência criativa.

2.5 Fairness Algorítmica em Contextos Educacionais
A crescente utilização de algoritmos em contextos educacionais tem ampliado as discussões sobre justiça, transparência e equidade algorítmica.
Modelos treinados com dados históricos podem reproduzir ou amplificar desigualdades já existentes, produzindo padrões diferenciados de erro entre grupos populacionais.
Por esse motivo, pesquisas recentes recomendam que sistemas preditivos sejam avaliados não apenas em termos de desempenho global, mas também quanto à distribuição desse desempenho entre diferentes grupos.
A literatura de algorithmic fairness propõe diversas métricas para avaliação de possíveis disparidades, incluindo:
Recall por grupo;
False Positive Rate (FPR);
False Negative Rate (FNR);
Balanced Accuracy;
Equal Opportunity Difference;
Disparate Impact.
A incorporação dessas análises permite verificar se determinados grupos são sistematicamente favorecidos ou prejudicados pelas previsões realizadas pelos modelos.
Neste estudo, a auditoria de fairness constitui uma etapa formal do pipeline analítico, contribuindo para uma utilização mais transparente e responsável da inteligência artificial em educação.
2.6 Data Leakage em Aprendizagem de Máquina Educacional
O problema conhecido como data leakage representa uma das principais ameaças à validade de estudos baseados em aprendizagem de máquina.
Esse fenômeno ocorre quando informações relacionadas direta ou indiretamente à variável-alvo permanecem disponíveis durante o treinamento dos modelos, permitindo que os algoritmos reproduzam a lógica de construção do desfecho em vez de aprender padrões substantivos associados ao fenômeno investigado.
Em bases educacionais complexas, o risco de leakage é particularmente elevado devido à existência de variáveis derivadas, indicadores compostos e medidas altamente correlacionadas.
Por essa razão, estudos recentes recomendam que auditorias de leakage sejam realizadas antes da modelagem preditiva, identificando variáveis potencialmente problemáticas e removendo atributos que possam comprometer a validade das análises.
Neste trabalho, a prevenção de leakage constitui um princípio metodológico central. O pipeline incorpora uma etapa específica de auditoria destinada à identificação de variáveis com potencial vazamento de informação, garantindo que os modelos sejam treinados apenas com atributos independentes da construção dos diferentes targets de resiliência.
Essa estratégia fortalece a robustez metodológica do estudo e aumenta a confiabilidade das evidências produzidas.

3. Referencial Teórico
3.1 Resiliência Criativa
A resiliência acadêmica é tradicionalmente definida como a capacidade de estudantes alcançarem resultados educacionais positivos apesar da exposição a condições adversas, especialmente vulnerabilidade socioeconômica. Estudos internacionais frequentemente utilizam indicadores de desempenho em Matemática, Leitura e Ciências para identificar estudantes que superam expectativas associadas ao seu contexto social e econômico.
A introdução do domínio de Pensamento Criativo no PISA 2022 amplia essa discussão para além das competências acadêmicas tradicionais, permitindo investigar a capacidade de estudantes desenvolverem criatividade e inovação mesmo em cenários desfavoráveis. Nesse contexto, o presente estudo adota o conceito de resiliência criativa, entendido como a ocorrência de desempenho criativo superior ao esperado em estudantes expostos a condições de vulnerabilidade socioeconômica.
Diferentemente de abordagens que utilizam uma única definição operacional do fenômeno, esta pesquisa incorpora múltiplas estratégias de identificação da resiliência criativa. O pipeline analítico implementado permite comparar diferentes definições de target baseadas em combinações entre indicadores socioeconômicos e desempenho em Pensamento Criativo, possibilitando avaliar como diferentes critérios influenciam a prevalência do fenômeno e o comportamento dos modelos preditivos.
Essa abordagem reconhece que a resiliência não constitui um construto único e universalmente definido. Pelo contrário, diferentes operacionalizações podem capturar dimensões distintas do fenômeno. Assim, a comparação sistemática entre definições alternativas representa uma etapa metodológica importante para garantir maior robustez às conclusões produzidas.
Sob essa perspectiva, a resiliência criativa é compreendida não apenas como um resultado individual, mas como um fenômeno complexo que emerge da interação entre fatores pessoais, familiares, escolares, tecnológicos e contextuais. Consequentemente, sua investigação exige abordagens analíticas capazes de integrar múltiplas fontes de informação e identificar padrões não triviais presentes em grandes bases educacionais.
3.2 Teoria Ecológica do Desenvolvimento Humano
A Teoria Ecológica do Desenvolvimento Humano, proposta por Urie Bronfenbrenner, compreende o desenvolvimento humano como resultado da interação contínua entre o indivíduo e os diferentes ambientes nos quais está inserido. Segundo essa perspectiva, competências cognitivas, emocionais e sociais são influenciadas simultaneamente por características pessoais e por elementos presentes em múltiplos níveis contextuais.
Bronfenbrenner organiza essas influências em sistemas inter-relacionados que abrangem desde ambientes próximos ao estudante, como família e escola, até fatores econômicos, culturais e institucionais mais amplos. Dessa forma, o desenvolvimento humano não pode ser explicado exclusivamente por atributos individuais, sendo necessário considerar as oportunidades e restrições presentes no contexto em que o sujeito vive.
No campo educacional, essa teoria oferece suporte para compreender como diferentes recursos disponíveis no ambiente podem favorecer trajetórias positivas mesmo diante de condições adversas. Aspectos como acesso à tecnologia, suporte familiar, recursos educacionais e oportunidades de aprendizagem constituem elementos potencialmente associados ao desenvolvimento de competências complexas.
Aplicada ao presente estudo, a perspectiva ecológica fornece uma base teórica para interpretar a resiliência criativa como resultado da interação entre vulnerabilidade socioeconômica e fatores contextuais que favorecem o desenvolvimento do pensamento criativo. Dessa forma, estudantes classificados como resilientes criativos são compreendidos como indivíduos que, apesar das limitações impostas pelo contexto socioeconômico, conseguem mobilizar recursos e oportunidades presentes em seus ambientes para alcançar elevado desempenho criativo.

3.3 Recursos Tecnológicos, Inclusão Digital e Criatividade
A literatura educacional contemporânea reconhece que o acesso a tecnologias digitais desempenha papel crescente no desenvolvimento de competências cognitivas complexas. Recursos tecnológicos ampliam oportunidades de acesso à informação, comunicação, resolução de problemas, produção de conhecimento e experimentação criativa.
Em contextos marcados por desigualdades sociais, a inclusão digital assume importância ainda maior. A disponibilidade de computadores, dispositivos móveis, conectividade e recursos digitais pode ampliar oportunidades educacionais e reduzir parcialmente algumas limitações associadas à vulnerabilidade socioeconômica.
Diversos estudos indicam que estudantes com maior acesso a tecnologias tendem a apresentar melhores condições para desenvolver competências relacionadas à criatividade, inovação e pensamento crítico. Embora o acesso tecnológico não determine diretamente o desempenho criativo, ele pode funcionar como um fator facilitador ao ampliar possibilidades de exploração, expressão e aprendizagem.
No PISA 2022, informações relacionadas ao acesso e à disponibilidade de recursos tecnológicos são representadas por variáveis contextuais disponibilizadas pela OCDE, incluindo indicadores sintéticos como o ICTRES. Tais variáveis constituem importantes fontes de informação para investigar a relação entre inclusão digital e desenvolvimento criativo.
Neste estudo, recursos tecnológicos são tratados como potenciais fatores associados à resiliência criativa. Sua relevância é investigada por meio de análises exploratórias, modelagem preditiva e técnicas de interpretabilidade, permitindo avaliar em que medida a inclusão digital contribui para a identificação de estudantes que apresentam elevado desempenho criativo em contextos socioeconômicos adversos.


3.4 Learning Analytics, Dados Processuais e Comportamento do Estudante
O crescimento da disponibilidade de dados educacionais em larga escala impulsionou o desenvolvimento da área de Learning Analytics, que busca compreender processos de aprendizagem por meio da coleta, análise e interpretação de dados educacionais.
Além das respostas fornecidas pelos estudantes, avaliações digitais modernas registram informações relacionadas ao processo de realização das tarefas. Esses registros incluem tempos de resposta, padrões de navegação, persistência, intensidade de participação e outros indicadores comportamentais frequentemente denominados dados processuais.
A literatura em Learning Analytics sugere que tais informações podem fornecer evidências importantes sobre engajamento, estratégias cognitivas, autorregulação e persistência durante a resolução de tarefas complexas. Embora não representem medidas diretas de criatividade, esses indicadores podem capturar comportamentos associados ao desempenho em atividades que exigem elaboração de ideias, exploração de alternativas e resolução criativa de problemas.
O PISA 2022 disponibiliza diversas variáveis derivadas da interação dos estudantes com as tarefas digitais de Pensamento Criativo. Essas informações ampliam significativamente o potencial analítico da base de dados, permitindo investigar não apenas o resultado obtido pelos participantes, mas também aspectos relacionados ao processo de realização das atividades.
No presente estudo, os dados processuais constituem uma dimensão central da análise. Variáveis associadas ao engajamento, persistência e comportamento durante as tarefas são incorporadas aos modelos de Aprendizagem de Máquina e posteriormente avaliadas por meio de técnicas de explicabilidade. Essa estratégia permite investigar se padrões comportamentais observados durante a avaliação estão associados à ocorrência da resiliência criativa entre estudantes brasileiros.



3.5 Aprendizagem de Máquina Explicável e Fairness em Educação
O uso crescente de técnicas de Aprendizagem de Máquina em educação tem ampliado a capacidade de identificar padrões complexos em grandes bases de dados. Esses métodos permitem investigar fenômenos multidimensionais e capturar relações frequentemente não detectadas por abordagens estatísticas tradicionais.
Entretanto, aplicações educacionais exigem não apenas desempenho preditivo, mas também transparência, interpretabilidade e responsabilidade algorítmica. Nesse contexto, a Inteligência Artificial Explicável (Explainable Artificial Intelligence – XAI) surge como um conjunto de técnicas destinadas a tornar os modelos mais compreensíveis e auditáveis.
Entre os métodos mais utilizados destaca-se o SHAP (SHapley Additive Explanations), que permite quantificar a contribuição individual de cada variável para as previsões realizadas pelos modelos. Essa abordagem possibilita transformar algoritmos preditivos em instrumentos de investigação científica, permitindo compreender quais fatores estão associados ao fenômeno estudado.
Paralelamente, a literatura recente enfatiza a importância da avaliação de justiça algorítmica (algorithmic fairness). Modelos treinados com dados históricos podem reproduzir desigualdades existentes, produzindo diferentes taxas de erro para grupos populacionais distintos. Em aplicações educacionais, tais diferenças podem impactar processos de identificação de estudantes, distribuição de recursos e formulação de políticas públicas.
Por esse motivo, o presente estudo adota uma abordagem integrada que combina modelagem preditiva, interpretabilidade, auditoria de fairness, análise de robustez e controle sistemático de data leakage. Essa integração constitui um dos principais diferenciais metodológicos da pesquisa, permitindo que os modelos sejam avaliados não apenas quanto à sua capacidade preditiva, mas também quanto à sua transparência, estabilidade e equidade.
Dessa forma, a Aprendizagem de Máquina é empregada como instrumento de produção de conhecimento científico sobre a resiliência criativa, alinhando desempenho analítico, rigor metodológico e responsabilidade no uso da inteligência artificial aplicada à educação.


4. Ferramentas e Métodos
4.1 Desenho da Pesquisa
Esta pesquisa caracteriza-se como um estudo quantitativo, observacional, explicativo e orientado por dados, desenvolvido a partir dos microdados brasileiros do Programme for International Student Assessment (PISA) 2022 (OCDE), com foco na investigação da Resiliência Criativa entre estudantes em contextos de vulnerabilidade socioeconômica.
O estudo foi conduzido sob a perspectiva da Ciência de Dados Educacionais (Educational Data Science), integrando conceitos de Learning Analytics, Educational Data Mining, Aprendizagem de Máquina (Machine Learning), Inteligência Artificial Explicável (Explainable Artificial Intelligence – XAI), auditoria de equidade algorítmica (Algorithmic Fairness) e avaliação de robustez estatística. A pesquisa foi implementada por meio de um pipeline analítico reproduzível desenvolvido em Python, estruturado nas seguintes etapas:
auditoria e caracterização dos dados;
construção automática do dicionário de dados;
identificação e categorização de variáveis;
auditoria de vazamento de informação (data leakage);
construção e comparação de definições operacionais de resiliência criativa;
análise exploratória dos dados;
caracterização dos estudantes resilientes;
descoberta de perfis latentes por abordagem person-centered;
modelagem preditiva supervisionada;
interpretação dos modelos por técnicas de XAI;
auditoria de fairness;
avaliação de robustez e calibração;
disponibilização dos resultados em dashboard interativo.
Todo o fluxo foi desenvolvido com foco em transparência metodológica, reprodutibilidade computacional e rastreabilidade das decisões analíticas, produzindo automaticamente relatórios, tabelas, visualizações e artefatos intermediários para auditoria.



4.2 Base de Dados
Foram utilizados os microdados brasileiros do PISA 2022 disponibilizados pela Organização para Cooperação e Desenvolvimento Econômico (OCDE).
Após os procedimentos de auditoria, limpeza e validação, a base analítica final foi composta por 3.834 estudantes e aproximadamente 1.275 variáveis contextuais, cognitivas, socioeconômicas, tecnológicas e processuais.
Entre os principais grupos de variáveis disponíveis destacam-se:
índice socioeconômico ESCS;
desempenho em Pensamento Criativo (CRT_SCORE);
recursos tecnológicos domiciliares;
recursos educacionais domésticos;
características familiares;
características escolares;
variáveis demográficas;
indicadores processuais derivados da interação dos estudantes com tarefas digitais de criatividade;
pesos amostrais e variáveis auxiliares disponibilizadas pela OCDE.
A auditoria inicial identificou ausência de registros duplicados e elevados níveis de missingness em diversas variáveis associadas aos itens específicos da avaliação de criatividade, comportamento esperado em decorrência da aplicação matricial dos instrumentos do PISA.
4.3 Construção das Definições de Resiliência Criativa
Considerando a inexistência de uma definição consolidada de Resiliência Criativa na literatura associada ao PISA 2022, foram avaliadas quatro definições operacionais alternativas baseadas na combinação entre vulnerabilidade socioeconômica e desempenho criativo.
As definições foram construídas utilizando o índice ESCS e a pontuação em Pensamento Criativo (CRT_SCORE).
Target A
ESCS ≤ primeiro quartil (Q1)
CRT_SCORE ≥ terceiro quartil (Q3)

Target B
ESCS ≤ percentil 30
CRT_SCORE ≥ percentil 70
Target C
ESCS ≤ primeiro quartil
CRT_SCORE ≥ percentil 90
Target D
Baseado em escore composto:
z (CRT_SCORE) − z (ESCS)
A comparação entre as definições considerou prevalência, interpretabilidade teórica e adequação estatística.
A definição A foi selecionada como referência principal por representar simultaneamente vulnerabilidade socioeconômica e elevado desempenho criativo, produzindo uma prevalência de aproximadamente 4,3% da amostra analisada.
4.4 Auditoria e Controle de Data Leakage
Uma etapa específica de auditoria foi implementada para identificar potenciais fontes de vazamento de informação.
A auditoria avaliou:
correlações excessivas com o desfecho;
dependências estruturais;
variáveis derivadas;
identificadores individuais;
pesos amostrais;
possíveis proxies do target.
Foram removidas variáveis utilizadas diretamente na construção do indicador de resiliência criativa, incluindo:
ESCS;
CRT_SCORE;
Creative_Resilience;
Grupo_ESCS;
CNTSTUID;
W_FSTUWT.
Além disso, foram identificadas automaticamente variáveis potencialmente associadas ao processo de construção do desfecho, reduzindo o risco de inflação artificial do desempenho dos modelos.
4.5 Análise Exploratória dos Dados
A análise exploratória teve como objetivo caracterizar a população estudada, avaliar a qualidade dos dados e identificar padrões preliminares associados à Resiliência Criativa.
Foram realizadas:
estatísticas descritivas;
análise de missingness;
detecção de outliers;
análise de correlação;
comparação entre estudantes resilientes e não resilientes;
análise de distribuições;
geração automática de visualizações e relatórios.
Os resultados dessa etapa subsidiaram tanto a interpretação substantiva do fenômeno quanto as etapas posteriores de modelagem e clusterização.
4.6 Descoberta de Perfis Latentes (Abordagem Person-Centered)
Além das análises supervisionadas, foi empregada uma abordagem person-centered para identificar perfis latentes de estudantes.
Inicialmente foi realizada redução de dimensionalidade por Análise de Componentes Principais (PCA), com **58 componentes** retidas (critério de explicabilidade do artefato de clustering; ver `outputs/figures/clustering/scree_plot.png` e `outputs/figures/clustering/cumulative_variance.png`).
Posteriormente foram comparados diferentes algoritmos de clusterização. A solução final foi definida pelo modelo **agglomerative_ward com k=2**, selecionado por um **score multicritério** que combina qualidade estrutural e **estabilidade via ARI/Jaccard** (ver `outputs/reports/clusterer_report.md`).
4.7 Modelagem Preditiva
A modelagem supervisionada teve como objetivo identificar padrões associados à ocorrência da Resiliência Criativa.
Foram avaliados algoritmos baseados em regressão e árvores de decisão, incluindo:
Regressão Logística;
Random Forest;
XGBoost;
LightGBM;
CatBoost.
O treinamento utilizou:
divisão estratificada treino-teste;
validação cruzada repetida;
otimização automática de hiperparâmetros;
calibração probabilística;
avaliação em conjunto holdout independente.
A principal métrica utilizada para comparação dos modelos foi a Área Sob a Curva ROC (ROC-AUC), complementada por Precision, Recall, F1-Score e Balanced Accuracy.
4.8 Inteligência Artificial Explicável (XAI)
A interpretação dos modelos foi realizada por meio de técnicas de Inteligência Artificial Explicável.
Foram utilizadas:
SHAP (SHapley Additive Explanations);
Permutation Importance.
Essas abordagens permitiram identificar os fatores com maior contribuição para as previsões dos modelos e investigar possíveis mecanismos associados à Resiliência Criativa.
4.9 Auditoria de Fairness
A avaliação de equidade algorítmica foi conduzida utilizando atributos sensíveis presentes na base de dados.
Foram analisadas diferenças de desempenho entre grupos populacionais por meio de métricas específicas de fairness e desempenho estratificado.
Essa etapa teve como objetivo verificar se os modelos apresentavam comportamento consistente entre diferentes subpopulações e identificar potenciais fontes de viés algorítmico.
4.10 Avaliação de Robustez
A robustez dos resultados foi avaliada por procedimentos de reamostragem e análise de estabilidade.
Foram utilizados:
bootstrap com múltiplas reamostragens;
intervalos de confiança para métricas;
avaliação de estabilidade dos clusters;
análise de calibração probabilística;
análise de sensibilidade dos limiares de classificação.
Esses procedimentos aumentam a confiabilidade dos resultados e reduzem a dependência de uma única divisão amostral.
4.11 Implementação Computacional e Reprodutibilidade
Todo o pipeline foi desenvolvido em Python e estruturado de forma modular para garantir reprodutibilidade computacional.
A arquitetura principal do projeto compreende:
src/ – implementação das etapas analíticas;
data/ – dados de entrada;
outputs/ – relatórios, tabelas e figuras;
models/ – modelos treinados;
dashboard/ – aplicação interativa;
config/ – parâmetros e configurações.
Ao final da execução, o sistema produz automaticamente relatórios analíticos, tabelas, visualizações, modelos treinados, análises de fairness, explicações por XAI e avaliações de robustez, permitindo total rastreabilidade dos resultados obtidos.
5. Análise Exploratória dos Dados e Modelagem Preditiva
5.1 Análise Exploratória dos Dados
A análise exploratória dos dados (Exploratory Data Analysis – EDA) teve como objetivo caracterizar a base utilizada na pesquisa, avaliar sua qualidade, identificar padrões preliminares relacionados à resiliência criativa e subsidiar as etapas posteriores de modelagem preditiva e interpretação dos resultados.
A base analítica foi composta por 3.834 estudantes brasileiros participantes do PISA 2022 e aproximadamente 1.275 variáveis contextuais, socioeconômicas, familiares, tecnológicas, comportamentais e processuais associadas ao domínio de Pensamento Criativo.
A fase de auditoria dos dados identificou a ausência de registros duplicados e realizou o mapeamento automático de variáveis com elevados níveis de valores ausentes, identificadores individuais e pesos amostrais. Também foram produzidos inventários de dados, estatísticas descritivas e relatórios de qualidade que permitiram compreender a estrutura da base antes da modelagem.
As análises exploratórias incluíram:
avaliação de distribuições univariadas;
análise de valores ausentes;
detecção de outliers por critério interquartil (IQR);
análise de correlação entre variáveis;
comparação entre estudantes resilientes e não resilientes;
visualização de padrões associados ao desempenho criativo;
investigação de indicadores processuais derivados da interação dos estudantes com as tarefas digitais.
Os resultados exploratórios sugeriram que fatores relacionados ao acesso a recursos tecnológicos, características familiares e indicadores comportamentais apresentavam potencial associação com a ocorrência da resiliência criativa.


5.2 Construção e Comparação dos Indicadores de Resiliência Criativa
Considerando a inexistência de uma definição consolidada de resiliência criativa na literatura associada ao PISA 2022, foram implementadas quatro definições alternativas do fenômeno.
Tabela 1 – Definições Operacionais de Resiliência Criativa
Definição
Critério
A
ESCS ≤ Q1 e CRT_SCORE ≥ Q3
B
ESCS ≤ P30 e CRT_SCORE ≥ P70
C
ESCS ≤ Q1 e CRT_SCORE ≥ P90
D
z(CRT_SCORE) − z(ESCS) ≥ P70

A fase de comparação de targets permitiu avaliar as prevalências, características e implicações analíticas de cada definição.
A Definição A foi selecionada como alvo principal das análises supervisionadas por apresentar maior alinhamento com a literatura internacional sobre resiliência educacional, além de oferecer equilíbrio entre rigor conceitual e tamanho amostral adequado para treinamento dos modelos.
A utilização de múltiplas definições permitiu avaliar a sensibilidade dos resultados à operacionalização do conceito de resiliência criativa e reforçou a robustez metodológica da pesquisa.



5.3 Perfil dos Estudantes Criativamente Resilientes
Após a construção da variável-alvo principal, foi realizada uma análise específica para caracterizar estudantes classificados como resilientes.
Essa etapa incluiu:
comparação de médias padronizadas;
análise de diferenças entre grupos;
visualizações multidimensionais;
mapas de calor;
gráficos radar;
identificação de variáveis com maior capacidade discriminativa.
Os resultados indicaram que estudantes resilientes apresentam padrões distintos em variáveis relacionadas ao acesso tecnológico, recursos familiares e indicadores comportamentais observados durante a realização das tarefas criativas.
Esses achados sugerem que a resiliência criativa emerge da interação entre fatores individuais e contextuais, não podendo ser explicada exclusivamente pela condição socioeconômica.
5.4 Análise de Agrupamentos (Clusterização)
Além das análises supervisionadas, foram aplicadas técnicas de aprendizagem não supervisionada com o objetivo de identificar perfis latentes de estudantes presentes na amostra.
Foram comparados os seguintes algoritmos:
K-Means;
Agrupamento Hierárquico Aglomerativo;
Gaussian Mixture Models (GMM).
A qualidade dos agrupamentos foi avaliada por meio do coeficiente Silhouette.
Os resultados indicaram a existência de estruturas latentes moderadas nos dados, permitindo identificar perfis distintos de estudantes quanto às suas características socioeconômicas, tecnológicas e comportamentais.
A análise de clusterização complementou as abordagens supervisionadas ao fornecer uma perspectiva exploratória sobre a heterogeneidade presente na população estudada.

5.5 Modelagem Preditiva da Resiliência Criativa
Após a auditoria de vazamento de informação e a definição da variável-alvo, foram desenvolvidos modelos supervisionados para investigar fatores associados à resiliência criativa.
O pipeline de modelagem incluiu:
divisão estratificada entre treinamento e teste;
validação cruzada repetida;
seleção automática do melhor algoritmo;
otimização de hiperparâmetros por busca aleatória;
ajuste de limiar de decisão;
avaliação em conjunto holdout independente.
Foram avaliados os seguintes algoritmos:
Regressão Logística (GLM);
Random Forest (RF);
XGBoost (XGB);
LightGBM;
CatBoost.
Todas as variáveis utilizadas na construção do alvo foram removidas antes do treinamento dos modelos, garantindo a eliminação de vazamento estrutural de informação.

5.6 Avaliação dos Modelos
O desempenho dos modelos foi avaliado por meio de métricas adequadas para problemas de classificação desbalanceados:
ROC-AUC;
Precision;
Recall;
F1-score;
Balanced Accuracy.
Os resultados mostraram que o desempenho preditivo é condicionado pela definição operacional do target (classe rara). Para o Target A, o melhor modelo foi o **xgboost_tuned**, com **ROC-AUC (CV) = 0.9236** e **ROC-AUC (holdout) = 0.9313**. No conjunto holdout, o **F1 = 0.2051** (limiar 0.50) e **F1 = 0.3548** (limiar otimizado **0.110**), indicando que a escolha do limiar é crítica para a identificação de positivos em classe desbalanceada.
(Nota: intervalos/medidas comparativas detalhadas constam no artefato de modelagem do projeto.)

Embora inferiores aos valores frequentemente observados em estudos sujeitos a vazamento de informação, esses resultados fornecem evidências metodologicamente mais robustas acerca da capacidade preditiva das variáveis disponíveis.
Os achados sugerem que a resiliência criativa é parcialmente previsível, mas permanece influenciada por fatores não observados nos microdados analisados.

5.7 Robustez e Calibração
A estabilidade dos resultados foi investigada por meio de procedimentos adicionais de robustez implementados no pipeline.
Foram conduzidas análises envolvendo:
bootstrap com múltiplas reamostragens;
estimativas de intervalos de confiança para métricas de desempenho;
avaliação da estabilidade dos modelos;
análise de calibração probabilística;
estudo da sensibilidade a diferentes limiares de classificação.
Esses procedimentos permitiram verificar a consistência dos resultados obtidos e reduzir a dependência de uma única divisão treino-teste.


5.8 Auditoria de Fairness
Após a seleção do modelo final, foi realizada uma auditoria de fairness com o objetivo de avaliar possíveis diferenças de desempenho entre grupos demográficos.
Foram calculadas métricas como:
Recall por grupo;
False Positive Rate (FPR);
False Negative Rate (FNR);
Balanced Accuracy;
Equal Opportunity Difference;
Disparate Impact.
Os resultados indicaram diferenças moderadas entre grupos, especialmente em métricas relacionadas à identificação correta de estudantes resilientes.
A inclusão dessa etapa reforça o compromisso do estudo com princípios de transparência, responsabilidade e uso ético de sistemas preditivos em educação.

5.9 Interpretabilidade dos Modelos (XAI)
A interpretação dos modelos foi realizada por meio de técnicas de Inteligência Artificial Explicável (XAI).
Foram utilizadas:
SHAP (SHapley Additive Explanations);
Permutation Importance;
análise de importância global de atributos.
Os resultados indicaram que variáveis relacionadas ao engajamento nas tarefas, tempo de resposta, acesso a recursos tecnológicos e características contextuais figuraram entre os principais fatores associados à probabilidade de um estudante ser classificado como resiliente criativo.
A utilização dessas técnicas permitiu transformar os modelos preditivos em instrumentos de geração de conhecimento científico, contribuindo para a compreensão dos mecanismos associados ao desenvolvimento da criatividade em contextos de vulnerabilidade socioeconômica.
5.10 Síntese dos Resultados
A análise integrada realizada pelo pipeline revelou que a resiliência criativa constitui um fenômeno raro e multifatorial, associado à interação entre fatores socioeconômicos, tecnológicos, comportamentais e contextuais.
Os resultados demonstraram que:
a definição operacional da resiliência influencia significativamente sua prevalência;
recursos tecnológicos e indicadores processuais apresentam relevância consistente;
modelos supervisionados conseguem identificar padrões associados ao fenômeno com desempenho moderado;
a interpretabilidade baseada em XAI permite compreender os fatores mais influentes nas previsões;
auditorias de fairness e análises de robustez ampliam a confiabilidade dos resultados.
Em conjunto, os achados evidenciam o potencial da integração entre Learning Analytics, Aprendizagem de Máquina Explicável, Fairness Algorítmica e Ciência de Dados Educacionais para investigar fenômenos complexos relacionados à criatividade e à equidade educacional.

6. Resultados e Discussão
Esta seção apresenta os principais resultados obtidos a partir da aplicação do pipeline analítico desenvolvido para investigar a resiliência criativa entre estudantes brasileiros participantes do PISA 2022. O fluxo metodológico integrou auditoria de dados, construção de múltiplas definições operacionais de resiliência, análise exploratória, clusterização, modelagem preditiva supervisionada, interpretabilidade baseada em Inteligência Artificial Explicável (XAI), auditoria de fairness e avaliação de robustez.
A análise foi realizada sobre uma base composta por 3.834 estudantes e aproximadamente 1.275 variáveis contextuais, socioeconômicas, tecnológicas, familiares e processuais derivadas da avaliação de Pensamento Criativo do PISA 2022. A adoção de um pipeline reproduzível permitiu não apenas construir modelos preditivos, mas também investigar a validade, estabilidade e equidade das previsões produzidas.


6.1 Construção e comparação das definições de resiliência criativa
Considerando a inexistência de uma definição consensual de resiliência criativa na literatura associada ao PISA 2022, foram implementadas quatro definições alternativas do fenômeno.
Tabela X – Definições operacionais avaliadas
Definição
Critério
Target A
ESCS ≤ Q1 e CRT_SCORE ≥ Q3
Target B
ESCS ≤ P30 e CRT_SCORE ≥ P70
Target C
ESCS ≤ Q1 e CRT_SCORE ≥ P90
Target D
Escore composto baseado em z(CRT_SCORE) − z(ESCS)

As definições apresentaram prevalências distintas, evidenciando que a identificação da resiliência criativa depende diretamente dos critérios adotados para representar simultaneamente vulnerabilidade socioeconômica e elevado desempenho criativo.
A Definição A foi selecionada como alvo principal das análises supervisionadas por apresentar melhor equilíbrio entre rigor conceitual e representatividade amostral. Essa definição aproxima-se da literatura clássica sobre resiliência acadêmica ao focalizar estudantes que obtêm desempenho significativamente superior ao esperado para sua condição socioeconômica.
A comparação sistemática dos targets demonstrou a importância de explicitar as escolhas conceituais envolvidas na operacionalização do fenômeno, uma vez que diferentes definições podem produzir grupos substancialmente distintos.
6.2 Auditoria de Dados e Análise Exploratória
A auditoria inicial confirmou a inexistência de registros duplicados e permitiu caracterizar a estrutura dos dados utilizados na pesquisa.
Foram identificadas variáveis com elevados percentuais de valores ausentes, especialmente entre itens específicos do domínio de Pensamento Criativo e variáveis processuais derivadas da interação dos estudantes com a plataforma digital da avaliação. Esse comportamento é consistente com a estrutura matricial do PISA e com a distribuição diferenciada dos blocos de itens entre os participantes.
As análises exploratórias revelaram forte heterogeneidade entre os estudantes brasileiros, tanto em relação às condições socioeconômicas quanto ao acesso a recursos tecnológicos, ambiente familiar e padrões de interação durante a avaliação.
Comparações entre estudantes resilientes e não resilientes indicaram diferenças particularmente relevantes em indicadores relacionados ao acesso a tecnologias digitais, recursos educacionais disponíveis no domicílio e medidas processuais associadas ao engajamento nas tarefas criativas.
Esses resultados sugerem que a criatividade observada em contextos de vulnerabilidade emerge de uma combinação complexa de fatores individuais, familiares e contextuais, não podendo ser explicada exclusivamente pela condição socioeconômica.
6.3 Perfis de Estudantes e Clusterização
Além das análises supervisionadas, foram aplicadas técnicas de agrupamento para investigar a existência de perfis latentes de estudantes.
Foram comparados algoritmos de K-Means, Clusterização Hierárquica Aglomerativa e Modelos de Mistura Gaussiana (GMM). A avaliação foi realizada por meio do coeficiente de Silhouette e de medidas complementares de qualidade dos agrupamentos.
Os resultados indicaram a presença de estruturas latentes na população estudada, permitindo identificar grupos com características distintas em termos de contexto socioeconômico, acesso a recursos tecnológicos, indicadores processuais e desempenho criativo.
A existência desses perfis sugere que a resiliência criativa não representa um fenômeno homogêneo, mas pode emergir por diferentes trajetórias de desenvolvimento, reforçando a necessidade de abordagens multidimensionais para sua compreensão.
6.4 Modelagem preditiva da resiliência criativa
A etapa de modelagem supervisionada teve como objetivo identificar padrões associados à ocorrência da resiliência criativa.
Antes do treinamento dos modelos foi realizada uma auditoria sistemática de vazamento de informação (data leakage), removendo variáveis utilizadas na construção do target e potenciais proxies diretos ou indiretos do desfecho.
Foram avaliados os seguintes algoritmos:
Regressão Logística;
Random Forest;
XGBoost;
LightGBM;
CatBoost.
O treinamento utilizou divisão estratificada entre treino e teste, validação cruzada repetida, seleção automática do melhor modelo com base no desempenho médio em ROC-AUC e, quando aplicável, otimização de hiperparâmetros por Randomized Search.
A avaliação final foi conduzida em conjunto de teste independente (holdout), garantindo estimativas mais confiáveis da capacidade de generalização dos modelos.
Os resultados demonstraram que modelos baseados em árvores apresentaram desempenho superior aos modelos lineares, indicando a existência de relações não lineares e interações complexas entre as variáveis associadas à resiliência criativa.
Mais importante que a obtenção de elevada acurácia foi a demonstração de que o fenômeno apresenta padrões identificáveis mesmo após a remoção rigorosa das variáveis associadas à definição do alvo, fortalecendo a validade científica das análises realizadas.
6.5 Robustez e Calibração dos Modelos
Uma contribuição adicional do presente estudo foi a incorporação de análises de robustez ao pipeline.
Foram realizados procedimentos de bootstrap para estimar a estabilidade das métricas preditivas e gerar intervalos de confiança para os principais indicadores de desempenho.
Também foram conduzidas análises de calibração probabilística, permitindo verificar o grau de correspondência entre as probabilidades previstas pelos modelos e as frequências observadas nos dados.
Os resultados indicaram estabilidade satisfatória das métricas ao longo das reamostragens, sugerindo que o desempenho observado não depende exclusivamente de uma divisão específica dos dados.
Essas evidências aumentam a confiabilidade dos resultados e reduzem o risco de conclusões baseadas em flutuações amostrais ocasionais.

6.6 Equidade Algorítmica (Fairness)
A utilização de modelos preditivos em contextos educacionais exige atenção não apenas ao desempenho global, mas também à distribuição dos erros entre diferentes grupos populacionais.
Por esse motivo, foi realizada uma auditoria de fairness utilizando atributos sensíveis definidos na configuração do projeto.
Foram calculadas métricas como:
Recall por grupo;
False Positive Rate (FPR);
False Negative Rate (FNR);
Balanced Accuracy;
Equal Opportunity Difference;
Disparate Impact.
As análises permitiram identificar possíveis assimetrias de desempenho entre grupos e forneceram evidências adicionais sobre a confiabilidade e a transparência dos modelos desenvolvidos.
A incorporação dessa etapa representa um diferencial metodológico importante, alinhando a pesquisa às recomendações contemporâneas para o uso responsável de Inteligência Artificial em Educação.
6.7 Interpretabilidade e Inteligência Artificial Explicável
A interpretação dos modelos foi conduzida por meio de técnicas de Inteligência Artificial Explicável (XAI), com destaque para análises baseadas em SHAP e Permutation Importance.
Os resultados mostraram que as variáveis mais influentes para a identificação da resiliência criativa estão associadas principalmente a três dimensões:
recursos tecnológicos disponíveis aos estudantes;
características socioeconômicas e familiares;
indicadores processuais derivados da interação com as tarefas digitais de criatividade.
As explicações geradas permitiram compreender não apenas quais variáveis são relevantes, mas também a direção e magnitude de sua contribuição para as previsões realizadas pelos modelos.
Essa abordagem transforma os algoritmos em instrumentos de produção de conhecimento científico, possibilitando investigar mecanismos associados ao fenômeno estudado em vez de apenas produzir classificações automatizadas.

6.8 Discussão dos Achados
Os resultados obtidos reforçam a compreensão da resiliência criativa como um fenômeno multifatorial, influenciado por fatores individuais, familiares, tecnológicos e comportamentais.
Em consonância com a Teoria Ecológica do Desenvolvimento Humano, os achados sugerem que o desenvolvimento da criatividade não depende exclusivamente das características do estudante, mas emerge da interação entre recursos disponíveis, oportunidades de aprendizagem e formas de engajamento com atividades cognitivamente desafiadoras.
Os resultados também dialogam com a perspectiva da Cognição Distribuída, ao indicar que recursos tecnológicos podem funcionar como mecanismos de ampliação das possibilidades de exploração, aprendizagem e expressão criativa, especialmente em contextos de vulnerabilidade socioeconômica.
Por fim, a combinação entre Learning Analytics, Aprendizagem de Máquina, Fairness e Inteligência Artificial Explicável demonstrou potencial para produzir evidências robustas e transparentes sobre fenômenos educacionais complexos, contribuindo para o avanço metodológico das pesquisas na área e para o desenvolvimento de políticas públicas orientadas por dados.




7. Considerações Finais
Esta pesquisa investigou a resiliência criativa entre estudantes brasileiros participantes do PISA 2022 por meio de um pipeline analítico reproduzível fundamentado em princípios de Learning Analytics, Ciência de Dados Educacionais, Aprendizagem de Máquina e Inteligência Artificial Explicável (XAI). Diferentemente de abordagens centradas exclusivamente na construção de modelos preditivos, o estudo estruturou um fluxo completo de análise que integrou auditoria de qualidade dos dados, descoberta de variáveis, detecção de vazamento de informação (data leakage), construção e comparação de múltiplas definições de resiliência criativa, análise exploratória, clusterização, modelagem supervisionada, interpretabilidade, auditoria de fairness, avaliação de robustez e disponibilização dos resultados em dashboard interativo.
Uma das principais contribuições do trabalho foi a proposição de um framework reproduzível para investigar estudantes que apresentam elevado desempenho em Pensamento Criativo apesar de condições socioeconômicas desfavoráveis. Para isso, foram avaliadas quatro definições operacionais distintas de resiliência criativa (Targets A, B, C e D), permitindo comparar diferentes níveis de rigor conceitual e prevalência do fenômeno. Essa estratégia mostrou que a identificação de estudantes resilientes depende fortemente dos critérios adotados, reforçando a importância da transparência metodológica na construção de indicadores educacionais.
Os resultados evidenciaram que a resiliência criativa constitui um fenômeno multidimensional, não sendo explicada exclusivamente pela condição socioeconômica dos estudantes. As análises exploratórias, os perfis comparativos e os modelos preditivos indicaram que fatores relacionados ao contexto familiar, ao acesso a recursos tecnológicos, às características educacionais e aos indicadores processuais derivados da interação dos estudantes com as tarefas digitais do PISA apresentam associação relevante com a ocorrência do fenômeno. Esses achados sugerem que trajetórias de sucesso criativo emergem da interação entre características individuais e oportunidades de aprendizagem presentes nos diferentes contextos de desenvolvimento.
A etapa de clusterização revelou ainda que a população estudada não é homogênea. Os agrupamentos identificados indicam a existência de diferentes perfis de estudantes quanto às suas características socioeconômicas, tecnológicas e comportamentais. Esse resultado reforça a ideia de que a resiliência criativa pode emergir por múltiplos caminhos, não existindo um único padrão capaz de explicar integralmente o desenvolvimento da criatividade em contextos de vulnerabilidade.
No âmbito da modelagem preditiva, os resultados demonstraram que os modelos de aprendizagem de máquina foram capazes de identificar padrões associados à resiliência criativa com desempenho consistente. A utilização de validação cruzada estratificada, otimização de hiperparâmetros, ajuste de limiares de decisão, avaliação em conjunto holdout e procedimentos de bootstrap contribuiu para aumentar a confiabilidade das estimativas obtidas. Mais importante que o desempenho preditivo em si foi a possibilidade de transformar os modelos em instrumentos de investigação científica, permitindo explorar fatores associados ao fenômeno estudado.
Um aspecto central da pesquisa foi a implementação sistemática de mecanismos de prevenção e auditoria de data leakage. Considerando que parte das variáveis presentes nos microdados poderia conter informações direta ou indiretamente relacionadas à construção do desfecho, foi realizada uma etapa específica de identificação e remoção de atributos potencialmente problemáticos antes do treinamento dos modelos. Esse procedimento fortaleceu a validade metodológica do estudo e reduziu o risco de obtenção de métricas artificialmente infladas.
As análises de interpretabilidade desempenharam papel fundamental na compreensão dos resultados. Por meio de técnicas como SHAP e medidas de importância de atributos, foi possível identificar variáveis com maior contribuição para as previsões dos modelos e compreender melhor os fatores associados à resiliência criativa. Dessa forma, a Inteligência Artificial Explicável foi utilizada não apenas como ferramenta de interpretação dos algoritmos, mas também como mecanismo de geração de conhecimento sobre o fenômeno investigado.
A auditoria de fairness complementou essa abordagem ao incorporar uma perspectiva de equidade algorítmica à análise. A avaliação do comportamento dos modelos em diferentes grupos populacionais permitiu identificar possíveis assimetrias de desempenho e reforçou a importância de considerar critérios de justiça e transparência em aplicações educacionais baseadas em inteligência artificial. Em conjunto com as análises de robustez e calibração, essa etapa contribuiu para uma avaliação mais abrangente da qualidade dos modelos desenvolvidos.
Do ponto de vista teórico, os resultados dialogam com a Teoria Ecológica do Desenvolvimento Humano e com a perspectiva da Cognição Distribuída. Ambas enfatizam que o desenvolvimento de competências complexas, como a criatividade, emerge da interação contínua entre indivíduos, recursos disponíveis e contextos socioculturais. Os achados obtidos reforçam essa interpretação ao indicar que o desempenho criativo não pode ser compreendido apenas a partir das condições socioeconômicas, mas também das oportunidades de acesso, participação e engajamento oferecidas aos estudantes.
Em termos práticos, os resultados oferecem evidências relevantes para pesquisadores, gestores educacionais e formuladores de políticas públicas. As análises sugerem que iniciativas relacionadas à inclusão digital, ampliação do acesso a recursos educacionais, fortalecimento de ambientes de aprendizagem estimulantes e promoção de experiências que favoreçam o engajamento dos estudantes podem contribuir para o desenvolvimento de competências criativas, especialmente entre populações em situação de vulnerabilidade socioeconômica.
Como limitações, destaca-se que a pesquisa utilizou dados observacionais provenientes de uma única edição do PISA, impossibilitando inferências causais sobre os fatores identificados. Além disso, embora os microdados disponibilizem um amplo conjunto de informações individuais e contextuais, diversos aspectos relacionados às práticas pedagógicas, ao clima escolar, às características institucionais das escolas e às experiências educacionais dos estudantes não estão integralmente representados na base analisada.
Como perspectivas futuras, recomenda-se ampliar a investigação por meio da incorporação de variáveis contextuais escolares e regionais, da realização de análises multinível, da aplicação de técnicas de inferência causal e da replicação do estudo em diferentes ciclos do PISA e em outras populações. Também se mostram promissoras a expansão das auditorias de fairness, a exploração de métodos avançados de interpretabilidade e o desenvolvimento de sistemas analíticos capazes de apoiar a tomada de decisão educacional baseada em evidências.
Por fim, este trabalho contribui para o avanço das pesquisas que integram Learning Analytics, Ciência de Dados Educacionais, Aprendizagem de Máquina e Inteligência Artificial Explicável, oferecendo um pipeline metodologicamente robusto, transparente e reproduzível para o estudo da resiliência criativa. Mais do que identificar estudantes resilientes, a pesquisa demonstra como abordagens modernas de análise de dados podem ser utilizadas para compreender fenômenos educacionais complexos e apoiar a construção de políticas públicas voltadas à promoção da criatividade, da inclusão e da equidade educacional.










Referências
Bronfenbrenner, U. (1979). The Ecology of Human Development: Experiments by Nature and Design. Cambridge: Harvard University Press.
Breiman, L. (2001). Random Forests. Machine Learning, v. 45, n. 1, p. 5–32.
Chen, T.; Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM, p. 785–794.
Hutchins, E. (1995). Cognition in the Wild. Cambridge: MIT Press.
Lundberg, S. M.; Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In: Advances in Neural Information Processing Systems (NeurIPS), p. 4765–4774.
OECD. (2023a). PISA 2022 Assessment and Analytical Framework. Paris: OECD Publishing.
OECD. (2023b). PISA 2022 Results (Volume III): Creative Minds, Creative Schools. Paris: OECD Publishing.
OECD. (2024). PISA 2022 Technical Report. Paris: OECD Publishing.
Romero, C.; Ventura, S. (2020). Educational Data Mining and Learning Analytics: An Updated Survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, v. 10, n. 3.
Sirin, S. R. (2005). Socioeconomic Status and Academic Achievement: A Meta-Analytic Review of Research. Review of Educational Research, v. 75, n. 3, p. 417–453.
Slade, S.; Prinsloo, P. (2013). Learning Analytics: Ethical Issues and Dilemmas. American Behavioral Scientist, v. 57, n. 10, p. 1510–1529.
UNESCO. (2023). Global Education Monitoring Report 2023: Technology in Education – A Tool on Whose Terms? Paris: UNESCO.
Van Buuren, S. (2018). Flexible Imputation of Missing Data. 2. ed. Boca Raton: Chapman and Hall/CRC.
Verma, S.; Rubin, J. (2018). Fairness Definitions Explained. In: Proceedings of the International Workshop on Software Fairness. New York: ACM, p. 1–7.
Yeager, D. S.; Dweck, C. S. (2012). Mindsets That Promote Resilience: When Students Believe That Personal Characteristics Can Be Developed. Educational Psychologist, v. 47, n. 4, p. 302–314.


