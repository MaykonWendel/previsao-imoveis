# Previsão de aluguel de imóveis 🏠

Projeto de Machine Learning para estimar o valor do aluguel de imóveis brasileiros.

## Dados
- Fonte: OpenML
- Local: `data/raw/`
  
## Como obter os dados
```bash
python -c "from sklearn.datasets import fetch_openml; df = fetch_openml(data_id=42688, as_frame=True, parser='liac-arff').frame; df.to_csv('data/raw/imoveis.csv', index=False)"
```
