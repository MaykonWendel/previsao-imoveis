import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Nomes mais simples que os originais
COLUNAS_RENOMEAR = {
    "hoa_(BRL)": "hoa",
    "rent_amount_(BRL)": "aluguel",
    "property_tax_(BRL)": "iptu",
}
# Derivadas do alvo: usar no modelo seria vazamento de dados
COLUNAS_VAZAMENTO = ["total_(BRL)", "fire_insurance_(BRL)"]

ALVO = "aluguel"
NUM_BASE = ["area", "rooms", "bathroom", "parking_spaces", "floor"]
NUM_TAXAS = ["hoa", "iptu"]
CAT = ["city", "animal", "furniture"]

ANDAR_MAX = 50  # acima disso é erro de cadastro (o máximo no dado era 301)


def limpar_dados(df: pd.DataFrame, quantil: float = 0.999) -> pd.DataFrame:
    """Remove vazamento, duplicatas e outliers (erros de cadastro do scraping)."""
    df = df.drop(columns=COLUNAS_VAZAMENTO).rename(columns=COLUNAS_RENOMEAR)
    df = df.drop_duplicates()

    # Limites calculados uma vez, sobre os dados já sem duplicatas
    cols_corte = ["area", "hoa", "iptu", ALVO]
    limites = {c: df[c].quantile(quantil) for c in cols_corte}

    mascara = df["floor"] <= ANDAR_MAX
    for coluna, limite in limites.items():
        mascara &= df[coluna] <= limite

    return df[mascara].reset_index(drop=True)


def colunas_entrada(incluir_taxas: bool = False) -> list[str]:
    """Colunas que o modelo recebe como entrada."""
    num = NUM_BASE + (NUM_TAXAS if incluir_taxas else [])
    return num + CAT


def construir_preprocessador(incluir_taxas: bool = False) -> ColumnTransformer:
    """Padroniza numéricas e aplica one-hot nas categóricas."""
    num = NUM_BASE + (NUM_TAXAS if incluir_taxas else [])
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])