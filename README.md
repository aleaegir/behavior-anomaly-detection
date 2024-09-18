# Detecção de Anomalias Comportamentais em Vídeos de Vigilância para a CITYSEG

Este projeto de extensão tem como objetivo detectar comportamentos anômalos em vídeos de vigilância para a empresa CITYSEG, que atua no ramo de segurança eletrônica. Utilizamos técnicas de inteligência artificial para realizar a análise dos vídeos capturados em tempo real.

## Estrutura do Projeto

- `data/`: Contém os vídeos e frames utilizados para o treinamento e teste do modelo.
  - `raw/`: Dados brutos, vídeos originais.
  - `processed/`: Frames processados para análise.
- `models/`: Modelos de machine learning treinados.
- `notebooks/`: Notebooks com análise exploratória e treinamento do modelo.
- `scripts/`: Scripts principais do projeto.
  - `video_processing.py`: Converte vídeos em frames.
  - `anomaly_detection.py`: Detecta anomalias nos vídeos.
  - `train_model.py`: Treina o modelo de detecção de anomalias.
  - `app.py`: Aplicação web para visualização dos resultados.
- `tests/`: Contém os testes unitários e de integração.
- `static/` e `templates/`: Arquivos estáticos e templates para a aplicação web.

## Como Executar o Projeto

### 1. Clonar o Repositório

```bash
git clone https://github.com/aleaegir/projeto-anomalias-vigilancia.git
cd projeto-anomalias-vigilancia
