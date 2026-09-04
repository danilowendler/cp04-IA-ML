# CP4 - MLP + MLflow + API de Predição - IMDB

Projeto acadêmico de IA & ML para classificar o sentimento de avaliações do **IMDB Dataset of 50K Movie Reviews**, sem TensorFlow.

# Integrantes
Italo Caliari Silva - RM554758
Júlio César Ruiz Zequin - RM554676
Danilo Gronski Wendler - RM 556602
Pedro Henrique Muzel Santos - RM 555983
Vitor Montemor Ismael - RM 556027

## Arquitetura

`review bruta -> limpeza de HTML -> TF-IDF -> MLPClassifier -> positive/negative`

O pipeline completo de pré-processamento e MLP é registrado como modelo no MLflow. A API recupera o modelo vencedor e aplica exatamente o mesmo pré-processamento usado no treinamento.

## Requisitos

- Python 3.12
- Aproximadamente 2 GB livres para ambiente, banco e artifacts
- Dataset `IMDB Dataset.csv` na raiz do projeto

## Instalação no PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se a execução de scripts estiver bloqueada, use diretamente `.\.venv\Scripts\python.exe`.

## Treinamento

```powershell
python -m src.train
```

O script valida os dados, remove valores ausentes e reviews duplicadas, separa treino e teste com estratificação, executa cinco configurações e registra parâmetros, métricas, relatórios, matrizes de confusão e modelos no MLflow. O maior F1-score define o vencedor, cuja URI fica em `artifacts/best_model_uri.txt`.

O F1-score da classe positiva é o critério principal porque equilibra precision e recall. Accuracy, precision e recall são medidas complementares.

## Interface do MLflow

O treinamento utiliza `mlflow.db`. Abra a interface com o mesmo backend:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
```

Acesse `http://127.0.0.1:5000`, abra `CP4_IMDB_MLP`, selecione os cinco runs mais recentes e use **Compare**. Confira os parâmetros e as métricas `accuracy`, `precision`, `recall` e `f1`.

## API

Depois de concluir o treinamento:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

- Swagger: `http://127.0.0.1:8000/docs`
- Modelo: `GET http://127.0.0.1:8000/health`
- Predição: `POST http://127.0.0.1:8000/predict`

Entrada:

```json
{
  "review": "This movie was fantastic and very entertaining!"
}
```

Resposta:

```json
{
  "prediction": "positive",
  "probability": 0.98
}
```

## Testes

```powershell
python -m pytest -q
```

## Estrutura

```text
.
|-- IMDB Dataset.csv
|-- requirements.txt
|-- README.md
|-- VIDEO_ROTEIRO.md
|-- src/
|-- tests/
`-- artifacts/
```

Depois da nova execução, consulte `artifacts/experiment_results.csv` e `artifacts/best_model.json`. O roteiro do vídeo deve usar as novas métricas.
