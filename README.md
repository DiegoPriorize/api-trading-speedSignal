
# **Trade Signal API**

A **Trade Signal API** é uma API que gera sinais de compra, venda ou manutenção (HOLD) para ativos financeiros com base em candles OHLCV (Open, High, Low, Close, Volume). A API usa uma combinação de indicadores técnicos como **EMA**, **RSI**, **ATR**, entre outros, para gerar esses sinais.

Este **README** fornece informações detalhadas sobre como instalar, configurar, utilizar a API e entender os parâmetros de configuração para diferentes perfis de investidor.

---

## **Instalação**

### **Passo 1: Clonar o Repositório**

Se você ainda não tem o repositório clonado, clone-o para o seu ambiente local:

```bash
git clone https://github.com/seu-usuario/trade-signal-api.git
cd trade-signal-api
```

### **Passo 2: Instalar Dependências**

Com o repositório clonado, instale as dependências do projeto. Você pode fazer isso utilizando `pip`:

```bash
pip install -r requirements.txt
```

### **Passo 3: Rodando a API Localmente**

Após instalar as dependências, você pode rodar a API localmente utilizando o **Uvicorn**:

```bash
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`.

---

## **Dependências**

O arquivo `requirements.txt` contém todas as dependências necessárias para rodar a API:

```text
fastapi==0.95.2
pydantic==1.10.7
uvicorn==0.22.0
```

- **FastAPI**: Framework para construir a API.
- **Pydantic**: Para validação de dados de entrada e saída.
- **Uvicorn**: Servidor ASGI para executar a aplicação FastAPI.

---

## **Como Usar com Docker**

1. **Criação da Imagem Docker**

   No terminal, navegue até o diretório onde o `Dockerfile` está localizado e execute o seguinte comando para criar a imagem Docker:

   ```bash
   docker build -t trade-signal-api .
   ```

2. **Executando o Docker**

   Após a criação da imagem, execute o container com o seguinte comando:

   ```bash
   docker run -d -p 8000:8000 trade-signal-api
   ```

A API estará disponível em `http://localhost:8000`.

---

## **Exemplo de Uso**

### **Formato de Entrada**

A API recebe os dados no formato JSON, como mostrado abaixo. Isso inclui os **timestamps** (T), preços de **abertura (O)**, **fechamento (C)**, **máximas (H)**, **mínimas (I)** e **volume (V)**, seguidos dos parâmetros de estratégia.

```json
{
  "T": [1622494800, 1622495100, 1622495400, 1622495700, 1622496000],
  "O": ["100.50", "101.00", "100.80", "101.20", "101.50"],
  "C": ["101.00", "100.90", "101.10", "101.30", "101.00"],
  "H": ["101.50", "101.30", "101.20", "101.40", "101.60"],
  "I": ["100.40", "100.60", "100.50", "100.70", "100.90"],
  "V": ["1200", "1500", "1400", "1300", "1250"],
  "params": {
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "rsi_buy": 30.0,
    "rsi_sell": 70.0,
    "atr_period": 14,
    "atr_tp_mult": 2.0,
    "atr_sl_mult": 1.0,
    "min_volume": 0.0,
    "require_rsi_filter": false
  }
}
```

### **Formato de Saída**

A API retorna uma resposta em JSON, com o sinal de **compra (BUY)**, **venda (SELL)** ou **manutenção (HOLD)**, incluindo o **timestamp**, o **preço** de fechamento, **razão** e os valores de **take_profit** e **stop_loss**, caso aplicável.

Exemplo de resposta:

```json
{
  "signal": "BUY",
  "timestamp": 1622496000,
  "iso_time": "2021-05-31T12:00:00Z",
  "price": 101.00,
  "reason": "EMA9 cruzou acima da EMA21.",
  "take_profit": 103.00,
  "stop_loss": 99.00,
  "precision": 90.0
}
```

---

## **Como Usar no FlutterFlow**

Para utilizar esta API no **FlutterFlow**, siga os passos abaixo:

1. **Criar um Novo Recurso de API no FlutterFlow:**
   - Acesse a seção "API Calls" do FlutterFlow.
   - Crie um novo recurso de API com o método **POST** para o endpoint `/signal`.
   - Insira a URL do seu servidor (se estiver usando localmente, pode usar `http://localhost:8000/signal`).

2. **Configuração do Corpo da Requisição:**
   - No FlutterFlow, configure os parâmetros da requisição com o formato JSON conforme mostrado acima (com **T**, **O**, **C**, **H**, **I**, **V** e **params**).

3. **Lidar com a Resposta:**
   - Após a chamada, você pode manipular a resposta JSON no FlutterFlow para exibir o **sinal**, **preço**, **take profit**, **stop loss**, etc., de acordo com a lógica do seu app.

---

## **Estratégias de Trading Usadas**

A API utiliza várias **estratégias de trading** para gerar sinais com base em diferentes indicadores técnicos. As principais estratégias são:

1. **Cruzamento de EMAs**:
   - A **EMA rápida** (curto prazo) cruza acima da **EMA lenta** (longo prazo) gerando um sinal de **compra (BUY)**.
   - Quando a **EMA rápida** cruza abaixo da **EMA lenta**, um sinal de **venda (SELL)** é gerado.

2. **Filtro de RSI**:
   - O **RSI** (índice de força relativa) é utilizado para filtrar sinais. Se o RSI for superior a um valor de sobrecompra (ex: 70), o sinal de **compra** é evitado. Se for inferior a um valor de sobrevenda (ex: 30), o sinal de **venda** é evitado.

3. **ATR para Stop Loss e Take Profit**:
   - O **ATR** (média de intervalo verdadeiro) é utilizado para calcular a volatilidade do mercado. A distância do **Stop Loss (SL)** e **Take Profit (TP)** é ajustada com base no valor do ATR.

---

## **Descrição Detalhada dos Parâmetros**

### **`ema_fast` (9)**
   - **Descrição**: Período da **EMA rápida** (curto prazo).
   - **Quanto menor, mais sensível será a média.**
   - **Exemplo**: `ema_fast = 5` para um comportamento mais sensível.
   - **Exemplo de Uso**:
     - **Conservador**: `ema_fast = 9`
     - **Agressivo**: `ema_fast = 5`

### **`ema_slow` (21)**
   - **Descrição**: Período da **EMA lenta** (longo prazo).
   - **Quanto maior, mais suavizada será a média.**
   - **Exemplo**: `ema_slow = 50` para um comportamento mais suave.
   - **Exemplo de Uso**:
     - **Conservador**: `ema_slow = 30`
     - **Agressivo**: `ema_slow = 15`

### **`rsi_period` (14)**
   - **Descrição**: Período do cálculo do **RSI**.
   - **Quanto maior o valor, mais suavizado será o RSI.**
   - **Exemplo**: `rsi_period = 10` para uma resposta mais rápida.
   - **Exemplo de Uso**:
     - **Conservador**: `rsi_period = 20`
     - **Agressivo**: `rsi_period = 10`

### **`rsi_buy` (30.0)**
   - **Descrição**: Valor abaixo do qual o ativo é considerado sobrevendido e um sinal de **compra** pode ser gerado.
   - **Exemplo**: `rsi_buy = 25.0` para uma compra mais agressiva.
   - **Exemplo de Uso**:
     - **Conservador**: `rsi_buy = 40.0`
     - **Agressivo**: `rsi_buy = 25.0`

### **`rsi_sell` (70.0)**
   - **Descrição**: Valor do RSI acima do qual o ativo é considerado sobrecomprado e um sinal de **venda** pode ser gerado.
   - **Exemplo**: `rsi_sell = 75.0` para evitar compras quando o mercado está muito forte.
   - **Exemplo de Uso**:
     - **Conservador**: `rsi_sell = 65.0`
     - **Agressivo**: `rsi_sell = 75.0`

### **`atr_period` (14)**
   - **Descrição**: Período do cálculo do **ATR**.
   - **Quanto maior, mais longo o histórico usado para calcular a volatilidade.**
   - **Exemplo**: `atr_period = 10` para uma resposta mais rápida.
   - **Exemplo de Uso**:
     - **Conservador**: `atr_period = 20`
     - **Agressivo**: `atr_period = 10`

### **`atr_tp_mult` (2.0)**
   - **Descrição**: Multiplicador para definir o **Take Profit (TP)** baseado no ATR.
   - **Exemplo**: `atr_tp_mult = 3.0` para um TP mais distante, aumentando o risco.
   - **Exemplo de Uso**:
     - **Conservador**: `atr_tp_mult = 1.5`
     - **Agressivo**: `atr_tp_mult = 3.0`

### **`atr_sl_mult` (1.0)**
   - **Descrição**: Multiplicador para definir o **Stop Loss (SL)** baseado no ATR.
   - **Exemplo**: `atr_sl_mult = 0.5` para um SL mais próximo, diminuindo o risco.
   - **Exemplo de Uso**:
     - **Conservador**: `atr_sl_mult = 1.0`
     - **Agressivo**: `atr_sl_mult = 0.5`
