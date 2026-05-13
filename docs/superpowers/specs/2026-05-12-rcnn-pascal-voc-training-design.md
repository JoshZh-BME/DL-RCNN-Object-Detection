# RCNN Pascal VOC Object Detection – Tervezési Dokumentáció

**Dátum:** 2026-05-12
**Projekt:** Klasszikus RCNN neurális háló objektumdetektáláshoz Pascal VOC 2012 adathalmazon
**Környezet:** Google Colab notebook, kagglehub adathalmaz elérés

## Cél

Olyan RCNN modell létrehozása, tanítása és optimalizálása, amely képes objektumok detektálására és osztályozására a Pascal VOC 2012 adathalmazon. A projekt célja a klasszikus RCNN architektúra működésének bemutatása és teljesítményének kiértékelése.

## Architektúra

- **Típus:** Klasszikus RCNN (Girshick et al. 2014)
- **Backbone:** ResNet50 (ImageNet előtanított, utolsó osztályozó réteg nélkül)
- **Régió javaslat:** Selective Search algoritmus (opencv-python)
- **Osztályozó fej:** SVM (LinearSVC) – osztályonként egy bináris osztályozó
- **Bounding box regresszor:** Ridge regresszió – osztályonként
- **Kimeneti osztályok:** 20 Pascal VOC osztály + háttér

## Notebook struktúra (colab_train.ipynb)

A teljes implementáció egyetlen Colab notebookban, a következő szekciókkal:

### 1. Adathalmaz feltérképezése
- Kagglehub letöltés és kicsomagolás
- Osztálylista és osztályeloszlás vizsgálata
- Képméretek és bounding box statisztikák
- Vizualizáció: minta képek annotációkkal, hisztogramok

### 2. RCNN háló létrehozása
- ResNet50 backbone definiálása (torchvision-ból)
- Feature extractor: ResNet50 avgpool előtti rétegek
- ROI pooling: régiók átméretezése egységes méretre (224x224)
- Osztályozó fej (SVM) és bbox regresszor (Ridge) – sklearn alapú

### 3. Adatelőkészítés
- Pascal VOC XML annotációk parse-olása
- Selective Search régió generálás (~2000 régió képenként)
- IoU-alapú címkézés:
  - IoU > 0.5: pozitív minta (osztály-specifikus)
  - IoU < 0.3: negatív minta (háttér)
- Augmentáció: vízszintes tükrözés, színjitter
- Train/val szétválasztás

### 4. Tanítás és értékelés
- **Fázis 1:** CNN feature kinyerés minden régióra
- **Fázis 2:** SVM osztályozók tanítása
- **Fázis 3:** Bounding box regresszorok tanítása
- Kiértékelés: mAP (mean Average Precision), per-class AP

### 5. Optimalizálás
- Hiperparaméterek hangolása:
  - IoU küszöbértékek (0.4, 0.5, 0.6)
  - Selective Search régiók száma
  - SVM C paraméter
- Architektúra variánsok összehasonlítása

### 6. Adatmennyiség vizsgálat
- Tanítóhalmaz méretének variálása: 10%, 25%, 50%, 100%
- Osztályszám variációk: 5, 10, 20 osztály
- mAP vs. adatmennyiség grafikonok

### 7. Végeredmény dokumentálása
- Összefoglaló táblázatok
- Optimalizálás előtti/utáni összehasonlítás
- Következtetések és megfigyelések

### 8. Demo script
- Betanított modell betöltése
- Objektumdetektálás tetszőleges képen
- Predikciók vizualizációja (bounding boxok, osztálycímkék, konfidencia)

## Függőségek

```
kagglehub
torch
torchvision
opencv-python (Selective Search)
scikit-learn (SVM, Ridge)
numpy, matplotlib, PIL
xml.etree (VOC XML parse)
```

## Adatfolyam

```
Pascal VOC 2012 (kagglehub letöltés)
  → VOC XML parse (osztályok, bboxok kinyerése)
  → Selective Search régiók generálása
  → IoU párosítás (pozitív/negatív minta címkézés)
  → ResNet50 feature kinyerés (minden régióra)
  → SVM tanítás + bbox regresszor tanítás
  → mAP kiértékelés a validációs halmazon
```

## Megkötések

- A notebook Colab-ban futtatható, GPU runtime-mal (T4 elegendő)
- RésNet50 használata előre betöltött ImageNet súlyokkal
- A tanítás ideje nem haladhatja meg a ~2-3 órát
- Minden szekció végén ellenőrző kód (assert, sanity check)
