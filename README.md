# BPA Decision Support Tool (versimpeld)

Versimpelde versie van de BPA jaarlijkse beheer-tool, gericht op decision support.

## Starten (lokaal)

```powershell
pip install -r requirements.txt
streamlit run src/bpa_beheer_ui.py
```

## Tabbladen
- 📊 Overzicht
- ✏️ Subscripties aanpassen (override-editor; geen bulk snelle-actie)
- ➕ Component toevoegen
- 🗑️ Component verwijderen
- 📈 Historiek (basisvoorraad-historiek + haalbaarheid (Z, serviceniveau) + investering vs. subscripties)
- 💰 Kostenanalyse
- 🔢 Subscriptiedrempel
- 🏷️ Classificatie (gewichten aanpasbaar; sheet, selectiemethode (top X), aggregatie
  (geometrisch) en ArticleType-filter (critical, onbekend) liggen vast)

## Streamlit Cloud
Repo bevat `requirements.txt` op root-niveau. Bij het aanmaken van de app op
share.streamlit.io: main file path = `src/bpa_beheer_ui.py`.
