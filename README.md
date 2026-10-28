# Swing Score — Streamlit

## Executar localment (Windows)

1. Instal·la Python 3.11 o 3.12.
2. Obre una terminal a la carpeta `SwingScore_Streamlit`.
3. Executa:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit obrirà l'aplicació al navegador. Necessita connexió a internet per obtenir dades de Yahoo Finance.

## Publicar a Streamlit Community Cloud

1. Puja `app.py` i `requirements.txt` a un repositori GitHub.
2. A Streamlit Community Cloud, crea una app nova i selecciona aquest repositori.
3. Indica `app.py` com a fitxer principal.
4. Desplega l'app.

## Ús

- Pots introduir tickers manualment, separats per comes o línies.
- També pots pujar un CSV de Yahoo Finance amb una columna `Symbol`, `Ticker` o `Tickers`.
- Si omples els dos camps, l'app combina els tickers i elimina duplicats.
- El motor és el del notebook Swing Score FINAL11; no s'hi han canviat els pesos ni les fórmules de puntuació.
- Les dades i els resultats són informatius, no una recomanació d'inversió.
