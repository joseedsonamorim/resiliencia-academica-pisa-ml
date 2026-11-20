**UNIVERSIDADE FEDERAL RURAL DE PERNAMBUCO**
**PRÓ-REITORIA DE PESQUISA E PÓS-GRADUAÇÃO**
**PPG EM INFORMÁTICA APLICADA**

<br><br><br>

**RESILIÊNCIA CRIATIVA NO PISA 2022: UMA ABORDAGEM DE LEARNING ANALYTICS E INTELIGÊNCIA ARTIFICIAL EXPLICÁVEL PARA IDENTIFICAÇÃO DE FATORES ASSOCIADOS AO ALTO DESEMPENHO CRIATIVO EM CONTEXTOS DE VULNERABILIDADE SOCIOECONÔMICA**

<br><br><br>

ALUNO: José Edson Amorim Sebastião
ORIENTADOR: Rodrigo Lins Rodrigues

<br><br><br><br><br>

RECIFE-PE – ABRIL/2026

***

### Identificação do plano

**a) Nome do discente:** José Edson Amorim Sebastião.
**b) Bolsista Sim (X) Não ( ) Agência:** CAPES
**c) Nível (mestrado/doutorado):** Mestrado.
**d) Entrada (ano/semestre):** 2026/1.
**e) Orientador:** Rodrigo Lins Rodrigues.
**f) Co-orientadores:** Não possui.
**g) Projeto de pesquisa do(a) orientador(a) ao qual está vinculado o plano (projeto guarda-chuva):** Resiliência Criativa: Avaliação do Pensamento Criativo no PISA 2022.
**h) Área de concentração:** Informática Aplicada.
**i) Linha de pesquisa:** Tecnologias Educacionais e Sociais.

***

### 1 Introdução

Pesquisadores contemporâneos reconhecem a criatividade como uma das competências essenciais para a sociedade do século XXI, fundamental por impulsionar a inovação tecnológica e a resolução de problemas de alta complexidade em cenários adversos (MANGION; BALDACCHINO, 2025). Ciente da urgência global desta métrica, a Organização para Cooperação e Desenvolvimento Econômico (OCDE) inseriu, de maneira pioneira, o domínio de Pensamento Criativo no ciclo do *Programme for International Student Assessment* (PISA) em 2022. Esta inserção oficial ofertou à comunidade acadêmica uma medida padronizada internacional inédita sobre a ideação divergente de estudantes adolescentes (BARBOT; KAUFMAN, 2025; RUTKOWSKI; RUTKOWSKI, 2025).

Apesar deste avanço na avaliação global, a literatura sobre políticas de equidade e desigualdade educacional apresenta severas lacunas. Enquanto pesquisas extensas abordam amplamente a **resiliência acadêmica** em áreas determinísticas como leitura, matemática e ciências (ZHENG et al., 2023; GÓMEZ-TALAL et al., 2025), o fenômeno específico da *resiliência criativa* permanece esparsamente explorado (BURAKGAZI; REISS, 2025). Falta clareza empírica e modelagem quantitativa rigorosa sobre os motivos estruturais, psicológicos e contextuais que permitem a estudantes brasileiros socialmente vulneráveis transcenderem estatísticas de fracasso, superando restrições domiciliares para alcançar alto desempenho no Pensamento Criativo, fenômeno este que se consolida como uma falha na correlação material-educacional.

Observando o **Estado da Arte**, a intersecção de algoritmos preditivos com grandes bases como o PISA tem atraído a atenção de pesquisadores. Alvarez-Garcia, Arenas-Parra e Ibar-Alonso (2024) utilizaram análises de clusters para identificar perfis de estudantes, enquanto Huang et al. (2024) aplicaram o algoritmo *XGBoost* e a biblioteca *SHAP* para mapear fatores influentes no letramento matemático. Adicionalmente, Chu (2025) avaliou as configurações que relacionam a posse de Tecnologia de Informação e Comunicação (ICT) ao desempenho criativo através de análises de conjuntos difusos. No entanto, esses esforços falham sistemicamente em duas frentes: a imensa maioria prioriza a maximização predatória da acurácia preditiva em detrimento da interpretabilidade ética (*Fairness*) e, concomitantemente, ignoram a modelagem matemática correta da incerteza das notas do PISA, que são compostas por *Plausible Values*, tratando-as inadvertidamente como médias absolutas.

Diante do exposto e da lacuna conceitual apresentada, esta pesquisa estrutura seus **problemas de pesquisa** sobre quatro interrogações centrais (identificabilidade, fatores-chave, eficácia preditiva sem viés e auditoria de equidade): (a) É estatisticamente possível isolar e identificar estudantes criativamente resilientes por meio das topologias contextuais do PISA 2022?; (b) Quais construtos tecnológicos, processuais e atitudinais exibem associação de maior magnitude explicativa com a resiliência criativa no Brasil?; (c) Modelos avançados de *Machine Learning* interpretável (como o *XGBoost*) detêm capacidade preditiva superior para capturar a resiliência criativa sem sofrer de *data leakage* de variáveis espúrias?; e, finalmente, (d) Os modelos supervisionados revelam disparidades punitivas perante métricas de *Fairness* quando auditados por subgrupos demográficos vulneráveis (como o nível de escolaridade parental)?

Para dirimir estas indagações de forma reprodutível, este estudo de Mestrado propõe investigar a resiliência criativa no Brasil com extrema precisão, utilizando as sub-áreas de *Learning Analytics*, *Educational Data Mining* e Inteligência Artificial Explicável (XAI), provendo diretrizes sociais transparentes baseadas em dados (ERSOY et al., 2024).

***

### 2 Objetivos

#### 2.1 Objetivo Geral
Investigar as matrizes sistêmicas e preditivas associadas à resiliência criativa em estudantes brasileiros do PISA 2022, implementando *pipelines* de *Learning Analytics*, Aprendizagem de Máquina Interpretável e métricas de auditoria ética algorítmica.

#### 2.2 Objetivos Específicos
*   Construir uma definição operacional matemática de *Resiliência Criativa* baseada na interseção estrita entre vulnerabilidade socioeconômica extrema (quartil inferior) e alta proficiência criativa (quartil superior).
*   Desenvolver um pipeline reprodutível (limpeza, imputação e transformação) dos microdados brutos do PISA 2022 Brasil, blindado contra vazamento de dados futuros (*Data Leakage*).
*   Caracterizar padrões comportamentais e sociodemográficos entre resilientes e não-resilientes utilizando técnicas de inferência estatística não-supervisionada.
*   Treinar, calibrar e validar modelos preditivos de árvores de decisão em gradiente (XGBoost, Random Forest), com proteção cruzada em *Holdouts*, para a classificação robusta de discentes.
*   Desconstruir as predições de caixa preta e quantificar a importância de domínios tecnológicos, processuais e contextuais através de valores de Shapley (*Explainable AI*).
*   Auditar o comportamento estrutural dos modelos através de métricas formais de equidade (*Algorithmic Fairness*) e calibrar erros de variação populacional usando pesos complexos da OCDE (*Fay's BRR*).

***

### 3 Material e Métodos

O delineamento desta pesquisa configura-se como de natureza quantitativa, com finalidades exploratórias e correlacionais-preditivas, alicerçado fundamentalmente na intersecção entre o *Educational Data Mining* e o *Learning Analytics*. Para justificar as amarrações do modelo algorítmico, a investigação apóia-se no referencial teórico da **Teoria Ecológica do Desenvolvimento Humano**, elaborada por Urie Bronfenbrenner. Esta lente teórica dita que a cognição do aluno emerge da sobreposição de mesossistemas (clima escolar e acesso a infraestrutura tecnológica) compensando as privações dos microssistemas (ambiente familiar e pobreza extrema), validando empiricamente a seleção do conjunto de variáveis da base para o algoritmo (BURAKGAZI; REISS, 2025; WEI et al., 2025).

#### 3.1 Coleta e Base de Dados (PISA 2022)
A matriz de dados originar-se-á da extração oficial dos microdados internacionais do PISA 2022 (liberados abertamente no repositório digital da OCDE), delimitados aos estudantes da unidade amostral do Brasil submetidos à prova de formato eletrônico baseada em computador (*Computer-Based Assessment* - CBA). O processo de montagem consolidará a base cognitiva (contendo os 10 *Plausible Values* de criatividade) aos questionários sociodemográficos através do identificador unitário único (`CNTSTUID`). O rol de dimensões avaliadas contemplará o Índice Socioeconômico e Cultural (*ESCS*), indicadores diretos de acesso e proficiência tecnológica (*ICT*), métricas de engajamento em testes (*process data*) e a literacia fundacional (Leitura e Ciências).

#### 3.2 Definição Matemática de Resiliência Criativa (Filtro e Target)
O fenômeno estudado será destilado utilizando uma clivagem inter-quartis rigorosa. A "Resiliência Criativa" — a classe alvo positiva do algoritmo — será ativada se, e somente se, o estudante brasileiro habitar simultaneamente as fatias marginais do desamparo e do triunfo: pertencer à faixa populacional ≤ 25% mais pobre no índice contínuo de status econômico (ESCS Q1) e, concomitantemente, pontuar ≥ 75% da distribuição do país nos resultados de criatividade (CRT Q4). Estratégias em percentis evitam os limites arbitrários e reduzem distorções por dependência de curvas normais puras, referendando o modelo clássico na literatura analítica educacional (ZHENG et al., 2023).

#### 3.3 Tratamento de Dados e Profilaxia de Leakage
A engenharia dos atributos (`features`) seguirá diretrizes de esterilização científica contra vazamentos (*Data Leakage Control*). Antes de inferir as predições espaciais, o algoritmo removerá cirurgicamente quaisquer dimensões da base vinculadas por identidade direta à construção do escore de criatividade ou do ESCS, evitando associações tautológicas e óbvias. Com tratamento estatístico para perdas não aleatórias, os hiperparâmetros serão expostos apenas ao conjunto de Treinamento segregado (80%), blindando o conjunto de Teste Cego (20%) para validação honesta da capacidade de generalização das funções estocásticas.

#### 3.4 Modelagem Preditiva e Análise Algorítmica
O arsenal preditivo investigará abordagens estatísticas densas, encabeçadas pela *Regressão Logística* e superadas por ensambles matriciais como *Random Forest* e *eXtreme Gradient Boosting* (*XGBoost*). Para dirimir o ruído da incerteza imposto pela Teoria de Resposta ao Item da OCDE, o sistema inferencial instanciará modelos simultâneos mapeando do *PV1 ao PV10* das notas criativas, orquestrando as sínteses por intermédio formal das **Regras de Rubin** (RUBIN, 1987). A avaliação algorítmica basear-se-á em testes robustos para falsos positivos em classes desbalanceadas: ROC-AUC, Precisão-Recall (PR-AUC), e pontuações do F1-Score. Erros populacionais de complexidade regional serão corrigidos aplicando-se a ponderação por repetição de balança de erros (*Fay's BRR*) provida no banco da OCDE.

#### 3.5 Explainable AI (XAI) e Governança de Fairness
Visando a ruptura da opacidade do Aprendizado de Máquina, o modelo XGBoost de melhor performance será devassado em seu subnúcleo interpretativo pela biblioteca de Teoria dos Jogos, a *SHAP (SHapley Additive exPlanations)*. Este passo elucidará, com granularidade a nível do estudante individual, o porquê de cada traço (como computadores domésticos ou atitudes extracurriculares) contribuir positiva ou negativamente na classificação (LI et al., 2025). Adicionalmente, empregar-se-ão matrizes éticas (Governança e Equidade Algorítmica) como *Disparate Impact* e *Equal Opportunity Difference*, testando ativamente o algoritmo para comprovar falhas discriminatórias sobre variáveis sensíveis, a exemplo de raça ou nível basal de educação materna (TAYLOR, 2025).

***

### 4 Hipóteses

*   **H1:** Estudantes classificados matematicamente como criativamente resilientes apresentam assinaturas sistêmicas nos dados processuais do PISA (ex.: persistência nos testes baseados em computador e engajamento comportamental) estatisticamente distintos, segregando-se do ruído de massa não resiliente em clusters de Aprendizado Não Supervisionado.
*   **H2:** O capital informacional, medido através de recursos tecnológicos e letramento cibernético ativo (*ICT*), detém associação preditiva altíssima com a propulsão da resiliência criativa, superando a influência financeira estática de rendimento (KAPOOR; TAGAT, 2025; LEZHNINA; KISMIHÓK, 2022).
*   **H3:** *Features* processuais oriundas dos painéis interativos virtuais (*process data* log files) e de atitude escolar atuam transversalmente, amplificando o poder preditivo do fenômeno de resiliência e suplementando as variáveis puramente demográficas.
*   **H4:** Modelos não-lineares e penalizados, como árvores impulsionadas por gradientes (XGBoost), atingem escores de Área Sob a Curva ROC-AUC de ordem superior (excelência) e comprovam estabilidade em bases de validação cruzada, imunes aos viesses do *overfitting* devido às regularizações *L2*.
*   **H5:** Diferenças demográficas sensíveis arraigadas na sociedade brasileira (gênero e ancestralidade escolar paterna) exibirão disparidade na taxa real-positiva de predição na auditoria do algoritmo, confirmando que a inteligência artificial expõe o viés sistêmico e não uma falha do sensor algorítmico, evidenciando o fosso da desigualdade educacional em contextos desamparados.

***

### 5 Contribuições Esperadas

#### Contribuições Científicas
*   **Fundação Teórica Inédita:** Proposição de uma definição operacional psicométrica e reprodutível de Resiliência Criativa atrelada à psicometria probabilística, preenchendo um profundo vazio metodológico no novo ciclo global do PISA (2022).
*   **Avanço na Integração:** Combinação transdisciplinar inédita que entrelaça o campo empírico da exclusão escolar à acurácia brutal do *Learning Analytics*, amarrado pelos preceitos da explicabilidade (*Explainable AI*) e da auditoria de equidade algorítmica (*Fairness*).

#### Contribuições Metodológicas
*   **Pipelines de Código Aberto:** Construção de arquiteturas modulares algorítmicas imunes a contaminações temporais (*Data Leakage*) para inferir classes raras de resiliência em grandes bases de educação.
*   **Inovação na Modelagem da Certeza:** Estruturação padronizada de rotinas estatísticas baseada nos princípios fundamentais de Rubin para múltiplos valores plausíveis conjugado ao erro de matrizes balanceadas de *Fay's BRR*.

#### Contribuições Tecnológicas
*   **Software Empacotado:** Confecção lógica de código de análise preditiva empacotada passível de registro perante o INPI, habilitando a reprodutibilidade.
*   **Dashboard Visual Interativo:** Desenvolvimento e projeção de painéis dinâmicos construídos para o gestor escolar, ilustrando os biomas e as pontes entre vulnerabilidade e ascensão (RAMOS; ARAYA, 2025).

#### Contribuições Sociais
*   **Políticas Públicas Precisas:** Descoberta de limiares probabilísticos exatos e perfis psicopedagógicos acionáveis, municiando entes federais do Estado com evidências não intuitivas acerca de infraestruturas (como letramento digital e suporte extraclasse) para políticas de resgate educacional eficientes.

***

### 6 Orçamento

A Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) financia integralmente esta pesquisa concedendo uma bolsa formativa sob nível stricto sensu de mestrado acadêmico. O suporte estipendial confere ao discente pesquisador a prerrogativa da dedicação em regime exclusivo aos estudos profundos. Adicionalmente, o Programa de Pós-Graduação em Informática Aplicada da UFRPE encabeça a alocação e manutenção de toda a infraestrutura física de base computacional para simulações e auditoria sistêmica. 

A operacionalização algorítmica não demandará compra de chaves proprietárias de fornecedores, posto que o ambiente computacional adotado baliza-se irrestritamente nas filosofias estritas do ecossistema *Open-Source* e linguagens livres (como Python e R) em conjunção com as bibliotecas públicas de *Data Science* e Inteligência Artificial (*Scikit-Learn, SHAP, Fairlearn*). A matéria-prima de suporte — os microdados e repositórios macro — é publicamente disseminada sem custos pela OCDE e órgãos de pesquisa de Paris. Por fim, possíveis custos transacionais englobando participações e submissões da tese em congressos e periódicos acadêmicos Qualis contarão com editais contínuos ou temporários de agências fomentadoras de pesquisa pernambucana e federais (FACEPE, CAPES). A pesquisa não possui contingenciamentos obstrutivos e atesta exequibilidade e sustentabilidade integrais.

***

### 7 Resultados Esperados

| Categoria | Número | Especificação |
| :--- | :---: | :--- |
| **Artigo científico** | 02 | Submissão de manuscrito hiper-estressado empiricamente em periódico internacional classificado com estratificação elevada (Qualis A) na interseção de EDM e Psicologia da Educação, bem como em anais de evento de vanguarda na área. |
| **Participação em congresso** | 01 | Divulgação oral/banner científico e sabatina das premissas e *frameworks* de arquitetura analítica e resultados exploratórios de perfis da resiliência em arena acadêmica qualificada. |
| **Patente ou equivalente** | 01 | Submissão formal de Registro de Software do ecossistema inteligente de extração relacional do pipeline e da biblioteca analítica de dashboarding, encapsulando metodologias reprodutíveis perante conselho local/nacional (ex: INPI). |
| **Livro ou capítulo** | 00 | Não aplicável. |
| **Dissertação de Mestrado** | 01 | Confecção massiva e documentação do documento formal de qualificação contínua e, por fim, o material probatório acadêmico rigoroso formatado para arguição pública e homologação do título no PPGIA. |
| **Participação em Editais** | 01 | Submissão ativa da proposta técnico-metodológica consolidada em chamadas públicas e arranjos institucionais para fomento investigativo ou subsídios a publicações abertas financiadas. |
| **Outras Produções** | 01 | Implantação de ambiente ou protótipo funcional visual focado no processamento local (ferramental como *Streamlit* ou *Shiny*) para permitir acesso rápido à demografia de predição do modelo aos gestores parceiros educacionais. |

***

### 8 Cronograma do Discente

| ATIVIDADES | 1º Sem | 2º Sem | 3º Sem | 4º Sem |
| :--- | :---: | :---: | :---: | :---: |
| **Créditos de disciplinas** | X | X | | |
| **Créditos de orientação** | X | X | X | X |
| **Pesquisa Bibliográfica** | X | X | X | |
| **Obtenção de dados** | X | X | | |
| **Análise de dados** | X | X | X | |
| **Proficiências** | | | X | X |
| **Revisão e redação da dissertação** | | | X | X |
| **Defesa da dissertação** | | | | X |

***

### Referências

ALVAREZ-GARCIA, M.; ARENAS-PARRA, M.; IBAR-ALONSO, R. Uncovering student profiles: An explainable cluster analysis approach to PISA 2022. *Computers & Education*, v. 223, p. 105166, 2024.

BARBOT, B.; KAUFMAN, J. C. PISA 2022 Creative Thinking Assessment: Opportunities, Challenges, and Cautions. *The Journal of Creative Behavior*, 2025.

BOURDIEU, P. The forms of capital. In: RICHARDSON, J. (Ed.). *Handbook of theory and research for the sociology of education*. New York: Greenwood, 1986. p. 241-258.

BURAKGAZI, S. G.; REISS, M. J. Exploring creative thinking skills in PISA: an ecological perspective on high-performing countries. *Frontiers in Psychology*, v. 16, 2025.

CHU, X. More ICT, more creativity? A fsQCA study on configurational pathways to students' creative thinking in PISA 2022. *International Journal of Educational Research*, v. 134, 2025.

ERSOY, Ö. et al. Stacking: An ensemble learning approach to predict student performance in PISA 2022. *Education and Information Technologies*, v. 30, p. 7753-7779, 2024.

GÓMEZ-TALAL, I.; BOTE-CURIEL, L.; ROJO-ÁLVAREZ, J. L. Interpretable Machine Learning Models for PISA Results in Mathematics. *IEEE Access*, v. 13, p. 27380-27397, 2025.

HUANG, Y. et al. Applying Machine Learning and SHAP Method to Identify Key Influences on Middle-School Students' Mathematics Literacy Performance. *Journal of Intelligence*, v. 12, n. 93, 2024.

KAPOOR, H.; TAGAT, A. The Importance of Context Awareness in PISA Creativity Data. *The Journal of Creative Behavior*, 2025.

LEZHNINA, O.; KISMIHÓK, G. Combining statistical and machine learning methods to explore German students' attitudes towards ICT in PISA. *International Journal of Research & Method in Education*, v. 45, n. 2, p. 180-199, 2022.

LI, J. et al. Student creative thinking in East Asia: The role of extracurricular activities and supportive environments. *Thinking Skills and Creativity*, v. 58, 2025.

MANGION, M.; BALDACCHINO, L. Commentary: Expanding the Scope of Creativity Assessment: Integrating Everyday Creativity to Address 21st Century Complexities. *The Journal of Creative Behavior*, 2025.

MARTIN, A. J.; MARSH, H. W. Academic resilience and its psychological and educational correlates: A construct validity approach. *Psychology in the Schools*, v. 43, n. 3, p. 267-281, 2006.

OECD. *PISA 2022 Results: Creative Minds, Creative Schools*. Paris: OECD Publishing, 2023.

RAMOS, J. H.; ARAYA, R. Do School Activities Foster Creative Thinking? An Analysis of PISA Results. *Education Sciences*, v. 15, n. 133, 2025.

RUBIN, D. B. *Multiple Imputation for Nonresponse in Surveys*. New York: Wiley, 1987.

RUTKOWSKI, L. et al. The Importance of Using Plausible Values in the Analysis of Large-Scale Assessment Data. *Educational Researcher*, v. 39, n. 2, 2010.

RUTKOWSKI, L.; RUTKOWSKI, D. Methodological Reflections on PISA'S Creative Thinking Assessment. *The Journal of Creative Behavior*, 2025.

TAYLOR, C. L. Best Practices for Leveraging PISA CT Data to Understand Gender Differences in Creative Thinking. *The Journal of Creative Behavior*, 2025.

WEI, Y. et al. Creative thinking and academic performance across cultures: Multilevel analyses based on PISA 2022. *British Journal of Educational Psychology*, 2025.

ZHENG, J. et al. Identifying key features of resilient students in digital reading: Insights from a machine learning approach. *Education and Information Technologies*, v. 29, p. 2277-2301, 2023.
