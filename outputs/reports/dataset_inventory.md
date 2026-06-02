#  RELATÓRIO DE AUDITORIA DO DATASET PISA BRASIL

**Data da auditoria:** 2026-05-31  
**Arquivo:** `pisa_brasil_estudo_limpo.csv`

---

##  DIMENSÕES DO DATASET

| Métrica | Valor |
|---------|-------|
| **Observações (Estudantes)** | 3,834 |
| **Variáveis (Colunas)** | 1275 |
| **Variáveis Numéricas** | 10 |
| **Variáveis Categóricas** | 1265 |

---

##  IDENTIFICAÇÃO DE VARIÁVEIS-CHAVE

###  Variável Dependente (Target)
- **Nome:** `Creative_Resilience`
- **Tipo:** Binária (0/1)
- **Posição:** Coluna 1273
- **Status:**  JÁ EXISTE NO DATASET

###  Variável de Desempenho em Criatividade
- **Nome:** `CRT_SCORE`
- **Tipo:** Numérica (contínua)
- **Posição:** Coluna 1272
- **Descrição:** Score agregado de Creative Thinking

###  Variável Socioeconômica
- **Nome:** `ESCS`
- **Tipo:** Numérica (índice padronizado)
- **Posição:** Coluna 1
- **Descrição:** Índice de Status Econômico, Social e Cultural (PISA)

###  Variáveis Demográficas
- `ST004D01T` (Gênero)
- `HISCED` (Educação parental)
- `CNTSTUID` (ID do Estudante)

###  Variáveis de Contexto
- `HOMEPOS` (Recursos de casa)
- `ICTRES` (Recursos de tecnologia)
- `W_FSTUWT` (Peso amostral)
- `Grupo_ESCS` (Quartis de ESCS)

###  Variáveis de Teste de Criatividade
- **Total:** 1265 itens/componentes
- **Padrão:** `CR[task_code]Q[item_number][type]`

---

##  ESTATÍSTICAS DESCRITIVAS

### ESCS (Índice Socioeconômico)
- **Mín:** -5.1083
- **Q1 (25%):** -1.6970
- **Mediana (50%):** -0.9422
- **Q3 (75%):** -0.1308
- **Máx:** 1.7973
- **Válidos:** 3,834

### CRT_SCORE (Desempenho em Creative Thinking)
- **Mín:** 0.0725
- **Q1 (25%):** 0.3371
- **Mediana (50%):** 0.4167
- **Q3 (75%):** 0.4925
- **Máx:** 0.9545
- **Válidos:** 3,834

### Creative_Resilience (Target)
- **0 (Não Resiliente):** 3,670 (95.72%)
- **1 (Resiliente):** 164 (4.28%)

### Status (Categorização Textual)
- `Demais estudantes`: 3,670 (95.72%)
- `Criativamente resiliente`: 164 (4.28%)

### Grupo_ESCS (Quartis)
- **Q1:** 959 (25.01%)
- **Q2:** 959 (25.01%)
- **Q3:** 958 (24.99%)
- **Q4:** 958 (24.99%)


---

##  VARIÁVEIS COM RISCO DE LEAKAGE

### Análise de Leakage

| Variável | Risco | Justificativa |
|----------|-------|--------------|
| `CRT_SCORE` |  CRÍTICO | Componente direto do target - NUNCA use |
| `Status` |  CRÍTICO | Derivada do target - NUNCA use |
| `Grupo_ESCS` |  ALTO | Derivada de ESCS - revisar conforme necessário |
| `CNTSTUID` |  MÉDIO | ID único - usar só para rastreamento |
| `W_FSTUWT` |  BAIXO | Peso amostral - usar em validação |

---

##  VARIÁVEIS A REMOVER DO PIPELINE

Não incluir no pré-processamento de features:
- `CNTSTUID` (ID)
- `W_FSTUWT` (Peso amostral - usar separadamente)
- `CRT_SCORE` (Componente do target)
- `Status` (Derivada do target)
- `Creative_Resilience` (Target - não é feature!)

---

##  VARIÁVEIS CANDIDATAS A FEATURES

### Principais (Nível Macro)
1. `ESCS` - Socioeconomia
2. `HOMEPOS` - Recursos de casa
3. `ICTRES` - Recursos tecnológicos
4. `HISCED` - Educação parental
5. `ST004D01T` - Gênero

### Complementares
- Todas as colunas `CR*` (itens de testes)

---

##  DEFINIÇÃO METODOLÓGICA DO TARGET

**Observação Crítica:** O dataset já possui `Creative_Resilience` construído.

Para validar a metodologia usada, verificar:
1.  Quartil Q1 de ESCS ≈ {escs_data[len(escs_data)//4]:.4f}
2.  Quartil Q3 de CRT_SCORE ≈ {crt_data[3*len(crt_data)//4]:.4f}
3. Contar quantos têm ESCS ≤ Q1 **E** CRT_SCORE ≥ Q3

Se {target_counts.get('1', 0)} = estudantes "Criativamente resilientes", a lógica foi aplicada corretamente.

---

##  PRÓXIMAS ETAPAS

1.  Auditoria de estrutura concluída
2.  Target validado
3. ⏳ Exploração de distribuições e correlações
4. ⏳ Engenharia de features
5. ⏳ Pré-processamento e pipeline de ML

*Relatório gerado em 2026-05-31*
