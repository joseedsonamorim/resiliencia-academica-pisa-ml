**UNIVERSIDADE FEDERAL RURAL DE PERNAMBUCO**
**PRÓ-REITORIA DE PESQUISA E PÓS-GRADUAÇÃO**
**PPG EM INFORMÁTICA APLICADA**

<br><br><br>

**RESILIÊNCIA CRIATIVA NO PISA 2022: UMA ABORDAGEM DE LEARNING ANALYTICS E INTELIGÊNCIA ARTIFICIAL EXPLICÁVEL PARA IDENTIFICAÇÃO DE FATORES ASSOCIADOS AO ALTO DESEMPENHO CRIATIVO EM CONTEXTOS DE VULNERABILIDADE SOCIOECONÔMICA**

<br><br><br>

ALUNO: José Edson Amorim Sebastião
ORIENTADOR: Rodrigo Lins Rodrigues
CO-ORIENTADOR: 

<br><br><br><br><br>

RECIFE-PE – ABRIL/2026

***

### Identificação do plano

**a) Nome do discente:** José Edson Amorim Sebastião.
**b) Bolsista Sim (X) Não ( ) Agência:** CAPES
**c) Nível (mestrado/doutorado):** Mestrado.
**d) Entrada (ano/semestre):** 2026/1.
**e) Orientador:** Rodrigo Lins Rodrigues.
**f) Co-orientadores:** 
**g) Projeto de pesquisa do(a) orientador(a) ao qual está vinculado o plano (projeto guarda-chuva):** Resiliência Criativa: Avaliação do Pensamento Criativo no PISA 2022.
**h) Área de concentração:** Informática Aplicada.
**i) Linha de pesquisa:** Tecnologias Educacionais e Sociais.

***

### 1 Introdução

Pesquisadores contemporâneos reconhecem a criatividade como uma das competências essenciais para a sociedade do século XXI, fundamental por impulsionar a inovação tecnológica e a resolução de problemas de alta complexidade em cenários adversos (MANGION; BALDACCHINO, 2025). Ciente da urgência global desta métrica, a Organização para Cooperação e Desenvolvimento Econômico (OCDE) inseriu, de maneira pioneira, o domínio de Pensamento Criativo no ciclo do *Programme for International Student Assessment* (PISA) em 2022 (BARBOT; KAUFMAN, 2025). Esta iniciativa insere-se intrinsecamente na linha de pesquisa de **Tecnologias Educacionais e Sociais**, visto que demanda arquiteturas de extração de conhecimento robustas para diagnosticar as fraturas e potenciais do capital humano da sociedade por meio de ferramentas analíticas computacionais.

Apesar deste avanço global, a literatura sobre políticas de equidade apresenta severas lacunas metodológicas e sociológicas. Enquanto pesquisas extensas abordam amplamente a resiliência acadêmica em áreas determinísticas como leitura e ciências (ZHENG et al., 2023; GÓMEZ-TALAL et al., 2025), o fenômeno específico da *resiliência criativa* permanece esparsamente explorado (BURAKGAZI; REISS, 2025). O **problema de pesquisa** subjacente a este plano reside na incapacidade sistêmica atual de prever e isolar os determinantes estruturais (capital tecnológico e apoio parental) que propiciam a um discente brasileiro superar a extrema pobreza (vulnerabilidade) para alcançar a elite do pensamento divergente, configurando uma exceção encorajadora à reprodução da desigualdade social.

O referencial da Teoria Ecológica do Desenvolvimento Humano, formulada por Urie Bronfenbrenner, postula que o desenvolvimento estudantil emerge do constante embate entre seus microssistemas (restrições do lar) e mesossistemas (clima tecnológico educacional). Na revisão do **Estado da Arte**, Alvarez-Garcia, Arenas-Parra e Ibar-Alonso (2024) demonstraram que análises de *clusters* são exímias em separar topologias socioeducacionais. Paralelamente, Huang et al. (2024) e Chu (2025) comprovaram a dominância preditiva das ferramentas baseadas em *Gradient Boosting* (como o algoritmo *XGBoost*) na literacia matemática, vinculando as infraestruturas de *Information and Communication Technology (ICT)* ao rendimento escolar. 

Contudo, este plano de pesquisa postula, como hipótese primária de intervenção e contribuição acadêmica, que os estudos vigentes incorrem em violações psicométricas profundas. Eles ignoram a essência probabilística de que o PISA é aferido por *Plausible Values*, operando indevidamente com médias lineares. Nosso documento endereça esta lacuna crítica através da fusão inédita entre o Aprendizado de Máquina Interpretável (*Explainable AI*) via Valores de Shapley e auditoria de equidade algorítmica (*Fairness*), suportados pela teoria clássica da imputação de valores incertos de Rubin e ponderação de variâncias amostrais complexas de *Fay's BRR* (RUTKOWSKI et al., 2010). Esta orquestração computacional destrinchará os preditores processuais, extraídos diretamente das interfaces de computador aplicadas no PISA, modelando o aluno que subverte a miséria material por meio da fluidez do pensamento inovador.

### 2 Objetivos 

#### 2.1 – Geral
Investigar as matrizes preditivas associadas à resiliência criativa em estudantes brasileiros do PISA 2022, implementando *pipelines* de *Learning Analytics*, Aprendizagem de Máquina Interpretável (*XAI*) e métricas de auditoria ética algorítmica.

#### 2.2 - Específicos
*   Construir uma definição operacional matemática de Resiliência Criativa baseada na interseção estrita entre vulnerabilidade socioeconômica extrema (quartil inferior) e alta proficiência criativa (quartil superior).
*   Desenvolver um *pipeline* reprodutível (limpeza, imputação e transformação) dos microdados brutos do PISA 2022 (Brasil), blindando o sistema contra vazamentos de dados futuros (*Data Leakage Control*).
*   Caracterizar padrões comportamentais latentes entre resilientes e não-resilientes utilizando técnicas de agrupamento não-supervisionado.
*   Treinar, calibrar e validar classificadores baseados em árvores de decisão em gradiente (*XGBoost*, *Random Forest*), com proteção cruzada em *Holdouts*.
*   Quantificar a importância das variáveis através de inteligência artificial explicável e auditar o comportamento dos modelos com matrizes algorítmicas de *Fairness*.

### 3 Metodologia

O delineamento desta pesquisa pauta-se no paradigma quantitativo, com finalidades exploratórias e correlacionais-preditivas, alicerçado no *framework Knowledge Discovery in Databases (KDD)*. O corpo teórico subjacente aos testes matemáticos apropria-se da **Teoria Ecológica de Bronfenbrenner** em conjugação com a tese contemporânea da **Segunda Exclusão Digital de Pierre Bourdieu** (BOURDIEU, 1986). Tais alicerces teóricos embasam a hipótese de que o capital cultural imaterial (letramento) e os recursos de interação utilitária (acesso doméstico à banda larga e familiaridade lógica com computadores) são preditores absolutos que sobrepujam as carências macroeconômicas na forja da resiliência discente.

A base e coleta de dados utilizará o banco oficial dos microdados internacionais do PISA 2022 liberado no portal da OCDE. A amostra limitará sua seleção demográfica aos respondentes da República Federativa do Brasil submetidos à prova de formato eletrônico computadorizado (*Computer-Based Assessment* - CBA). Para tanto, será orquestrada a extração massiva e o cruzamento direto dos arquivos brutos `.SAV` da base de características sociodemográficas familiares com os resultados apurados nos componentes de habilidade mental utilizando o vinculador serial `CNTSTUID`.

O fenômeno central (Target Binário) da Resiliência Criativa adotará modelagem por limites numéricos de quartis estatísticos: computar-se-á como amostra `Positiva=1` os jovens enquadrados concomitantemente no primeiro quartil de estrato socioeconômico (ESCS Q1) e no último quartil do índice avaliativo de pensamentos criativos (CRT Q4). Estratégias não-paramétricas em percentis reduzem sensivelmente distorções, evitam limiares de corte apriorísticos de viés geográfico e consolidam-se como o *modus operandi* rigoroso no campo (ZHENG et al., 2023).

O desenvolvimento das topologias no processamento preventivo contra contaminações de dados futuros será rigoroso. A imputação intermitente dos resíduos na matriz segregará antecipadamente o tecido matricial entre Treino (80%) e *Holdout* Restrito (20%). O tratamento inferencial será concebido a partir dos 10 *Plausible Values* de criatividade gerados no processamento log-verossimilhança da OCDE, empilhando 10 instâncias sucessivas de modelos XGBoost que, ao final, serão agregados mediante processamento das **Regras de Rubin** (RUBIN, 1987). Por corolário, o desvio de estimativa do censo escolar por dependências regionais de escolas será atenuado mediante as compensações matemáticas estipuladas nos 80 pesos replicados (Técnica de aproximação de variância de *Fay's BRR*).

Finalmente, os critérios de eficácia para validação englobarão as áreas sob as curvas Receiver Operating Characteristic (ROC-AUC) e Precision-Recall (PR-AUC), penalizando os falsos positivos comuns em amostras raras. O arcabouço da explicabilidade e transparência ética, fundamentais à área de *Learning Analytics*, destrinchará o peso exato destas arquiteturas pela matriz probabilística Teoria dos Jogos (*SHAP - SHapley Additive exPlanations*) (LI et al., 2025). As intersecções algorítmicas sensíveis também sofrerão auditorias sociológicas demográficas com as pontuações punitivas de igualdade por *Disparate Impact Ratio*.

### 4 Orçamento

A Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) financia esta pesquisa com fomento e suporte formativo a nível stricto sensu de mestrado, o qual assegura o arranjo logístico primordial à dedicação investigativa irrestrita ao escopo do presente projeto de dissertação.

O arranjo infraestrutural não demandará recursos volumosos complementares por ser operacionalizado em ambientes computacionais do próprio pesquisador discente junto aos laboratórios adjuntos de apoio cedidos pelo Programa de Pós-Graduação em Informática Aplicada da Universidade Federal Rural de Pernambuco (PPGIA/UFRPE). A base sistêmica é integralizada na cultura *Open-Source*, desobrigando pagamentos a plataformas de software proprietário de predição ao adotar ecossistemas das linguagens abertas Python (bibliotecas *Scikit-Learn, SHAP, Fairlearn*). A *database* do PISA de microdados é inteiramente pública e provida pela OCDE sem subscrições monetárias de acesso. Despesas residuais associadas a registros técnicos (software) e taxas de publicação de manuscritos Qualis gozam da previsibilidade de suporte advindo de editais contínuos ou auxílios operacionais da Pró-Reitoria, garantindo ampla robustez e plena sustentabilidade orçamentária do projeto de mestrado durante toda sua exequibilidade prevista de 24 meses.

### 5 Resultados Esperados

| Categoria | Número esperado |
| :--- | :--- |
| Artigo científico | 02 |
| Participação em congresso | 01 |
| Patente ou equivalente | 01 |
| Livro ou capítulo de livro | 00 |
| Dissertação de Mestrado | 01 |
| Tese de Doutorado | 00 |
| Participação em Editais | 01 |
| Outras (Desenvolvimento de painel interativo analítico) | 01 |

### 6. Cronograma do discente

| ATIVIDADES | 1º | 2º | 3º | 4º | 5º | 6º | 7º | 8º |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Créditos de disciplinas | X | X | | | | | | |
| Créditos de orientação | X | X | X | X | | | | |
| Pesquisa Bibliográfica | X | X | X | | | | | |
| Obtenção de dados | X | X | | | | | | |
| Análise de dados | X | X | X | | | | | |
| Proficiências (inglês/espanhol) | | | X | X | | | | |
| Exame de qualificação* | | | | | | | | |
| Revisão e redação da tese | | | X | X | | | | |
| Defesa da tese | | | | X | | | | |
* Só para doutorado

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
