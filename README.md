# KronosFlow AI

## Om projektet 💡
**KronosFlow AI** är en plattform för modellövervakning, klassificering av marknadsregimer och finansiell backtesting för aktier på den svenska OMXS30-listan.

Genom att kombinera avancerad feature engineering med övervakade maskininlärningsmodeller, ger systemet användare möjlighet att utvärdera prediktionsförmågan hos olika AI-arkitekturer på historisk aktiedata.

### Varför finns det? 🎯
Att bygga och deploya handelsstrategier baserade på maskininlärning kräver rigorös validering. Detta projekt skapades för att:
- **Brygga gapet** mellan prediktioner (t.ex. klassificering av prishorisonten Köp/Sälj/Behåll) och faktisk finansiell avkastning (genom simulering av portföljhandel).
- **Motverka overfitting** genom att erbjuda automatiserad Out-of-Sample-validering (de sista 180 dagarna hålls helt utanför träningen).
- **Visualisera prestanda** i realtid via en interaktiv instrumentpanel som visar klassificeringsmetrik (Precision, Recall, F1-Score, Confusion Matrices) sida vid sida med simulerade kapitalutvecklingskurvor.

### Vad kan man göra med det? 🛠️
- **Hämta & Lagra Data:** Ladda ner historisk pris- och volymdata för OMXS30 direkt via Yahoo Finance till en lokal SQLite-databas (`data/finance.db`).
- **Feature Engineering:** Beräkna tekniska indikatorer (RSI, MACD, Bollinger Bands) kombinerat med oövervakad maskininlärning (PCA för dimensionsreduktion och K-Means för att identifiera marknadsregimer).
- **Träna & Jämföra Modeller:** Träna en traditionell **Random Forest** och en djup **PyTorch LSTM med Attention** utrustad med Focal Loss för att hantera klassobalans.
- **Backtesta Strategier:** Simulera handel baserat på modellernas köp- och säljsignaler och mäta metrics såsom total avkastning, Buy & Hold-avkastning, vinstfrekvens och antal affärer.
- **Utforska EDA & Grafer:** Generera automatiskt distributions- och korrelationsgrafer samt granska modeller interaktivt i Streamlit-dashboarden.

### 📓 Projektresan (rekommenderad startpunkt)
[`notebooks/project_journey.ipynb`](notebooks/project_journey.ipynb) är den bästa platsen att börja på för att förstå projektet. Den är en körd, berättande genomgång av hela pipelinen — datarening, EDA, oövervakad inlärning, modellträning och modell-utvärdering — med riktiga diagram genererade mot projektets faktiska data och tränade modeller, samt löpande markeringar av de milstolpar, beslut och hinder som dök upp under arbetets gång.

---

## Projektstruktur 📂

```text
sommar-projekt/
├── config.py              # Centraliserade sökvägar och konfigurationsinställningar
├── requirements.txt       # Paketberoenden
├── data/
│   ├── finance.db         # Historisk prisdatabas (SQLite)
│   └── evaluation_report.json
├── src/
│   ├── ingestion.py       # Hämtar historisk data från Yahoo Finance
│   ├── preprocessing.py   # Rengör och förbereder träningsdata (Deduplicering & Pipeline-steg)
│   ├── features.py        # Feature engineering (RSI, MACD, Bollinger, PCA, K-Means)
│   ├── models.py          # Modellarkitekturer (Random Forest & LSTM + Focal Loss)
│   ├── evaluate.py        # Out-Of-Sample backtesting och klassificeringsrapportering
│   ├── eda.py             # Automatiserad EDA (sparar visualiseringar till outputs)
│   └── dashboard.py       # Streamlit-dashboard för interaktiv analys
├── outputs/               # Genererade artefakter och grafer
│   └── eda/               # EDA visualiseringar (RSI, Korrelation, Fördelning)
├── models/                # Sparade modelfiler (*.pkl, *.pth)
└── notebooks/
    └── project_journey.ipynb  # Berättande genomgång: milstolpar, beslut, hinder & alla diagram
```

## Hur man kör projektet 🚀

1. **Aktivera den virtuella miljön och installera beroenden**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   Vill du bara se projektet snabbt utan att köra något själv, öppna direkt
   [`notebooks/project_journey.ipynb`](notebooks/project_journey.ipynb) — den är redan körd och sparad
   med alla diagram och resultat:
   ```bash
   jupyter notebook notebooks/project_journey.ipynb
   ```

2. **Hämta rådata och fyll databasen**:
   ```bash
   python src/ingestion.py
   ```

3. **Generera automatiserad EDA**:
   ```bash
   python src/eda.py
   ```

4. **Träna modellerna (Random Forest & PyTorch LSTM)**:
   ```bash
   python src/models.py
   ```

5. **Utvärdera och kör finansiell backtesting**:
   ```bash
   python src/evaluate.py
   ```

6. **Starta webbdashboarden interaktivt**:
   ```bash
   streamlit run src/dashboard.py
   ```
