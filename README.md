# Como usar
Para instalar as dependências e ativar o ambiente virtual:
```bash
python3 -m venv .penv
source .penv/bin/activate
# requirements.txt foi gerado usando `pip freeze > requirements.txt`
pip install -r requirements.txt
```

## Padronizar as imagens
Após instalar as execute o programa
```bash
python3 segmentar.py
```

## (Opcional) Alterar técnica de segmentação
Por padrão, a segmentação está definida para **K-Means**, caso deseja segmentar por **Distância euclidiana**, basta alterar a variável `tipo_segmentacao` no inicio da função `main` do arquivo executável. (Requer um tempo maior de processamento)
