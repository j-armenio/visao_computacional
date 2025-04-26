# Como usar
Para instalar as depêndencias e ativar o ambiente virtual:
```bash
python3 -m venv .penv
source .penv/bin/activate
# requirements.txt foi gerado usando `pip freeze > requirements.txt`
pip install -r requirements.txt
```

## Padronizar as imagens
Após instalar as depêndencias
```bash
python3 padronizar.py
```