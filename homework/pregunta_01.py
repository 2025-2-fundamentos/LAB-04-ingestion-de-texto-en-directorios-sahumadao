import os
import zipfile
import pandas as pd

# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""



"""
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """
  


# Ruta base del proyecto (carpeta raíz)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas relevantes
FILES_DIR = os.path.join(BASE_DIR, "files")
INPUT_ZIP = os.path.join(FILES_DIR, "input.zip")
INPUT_DIR = os.path.join(FILES_DIR, "input")
OUTPUT_DIR = os.path.join(FILES_DIR, "output")


def _ensure_input_unzipped():
    """
    Descomprime input.zip si la carpeta input/ no existe.
    """
    if not os.path.exists(INPUT_DIR):
        print("Descomprimiendo input.zip...")
        with zipfile.ZipFile(INPUT_ZIP, "r") as z:
            z.extractall(FILES_DIR)


def _build_dataframe(split_path: str) -> pd.DataFrame:
    """
    Construye un dataframe a partir de:
    - files/input/train
    - files/input/test
    y sus subcarpetas:
    - neutral/
    - positive/
    - negative/
    """
    rows = []
    sentiments = ["neutral", "positive", "negative"]

    for sentiment in sentiments:
        sentiment_dir = os.path.join(split_path, sentiment)

        if not os.path.exists(sentiment_dir):
            raise FileNotFoundError(f"No existe la carpeta: {sentiment_dir}")

        for filename in os.listdir(sentiment_dir):
            file_path = os.path.join(sentiment_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            rows.append({
                "phrase": text,
                "target": sentiment
            })

    return pd.DataFrame(rows)


def pregunta_01():
    # 1. Garantizar que input.zip esté descomprimido
    _ensure_input_unzipped()

    # 2. Crear carpeta output si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Construir datasets
    train_df = _build_dataframe(os.path.join(INPUT_DIR, "train"))
    test_df = _build_dataframe(os.path.join(INPUT_DIR, "test"))

    # 4. Guardar CSV exactamente donde pytest los busca
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_dataset.csv"), index=False)