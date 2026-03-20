from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px

try:
    import streamlit as st
except ModuleNotFoundError:  # fallback para validação local sem Streamlit instalado
    class _DummySidebar:
        def subheader(self, *args, **kwargs):
            return None
        def multiselect(self, *args, **kwargs):
            return []
        def toggle(self, *args, **kwargs):
            return kwargs.get("value", False)
        def date_input(self, *args, **kwargs):
            return kwargs.get("value")
        def file_uploader(self, *args, **kwargs):
            return None
        def markdown(self, *args, **kwargs):
            return None
        def write(self, *args, **kwargs):
            return None

    class _DummySt:
        sidebar = _DummySidebar()
        def set_page_config(self, *args, **kwargs):
            return None
        def cache_data(self, show_spinner=False):
            def decorator(fn):
                return fn
            return decorator
    st = _DummySt()


st.set_page_config(
    page_title="Dashboard de Margem BI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Utilidades
# =========================

STATE_TO_REGION = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste", "PE": "Nordeste",
    "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

UF_COORDS = {
    "AC": (-9.97, -67.81), "AL": (-9.66, -35.74), "AP": (0.03, -51.05), "AM": (-3.10, -60.02),
    "BA": (-12.97, -38.50), "CE": (-3.73, -38.52), "DF": (-15.79, -47.88), "ES": (-20.32, -40.34),
    "GO": (-16.68, -49.25), "MA": (-2.53, -44.30), "MG": (-19.92, -43.94), "MS": (-20.47, -54.62),
    "MT": (-15.60, -56.10), "PA": (-1.45, -48.49), "PB": (-7.12, -34.86), "PE": (-8.05, -34.88),
    "PI": (-5.09, -42.80), "PR": (-25.43, -49.27), "RJ": (-22.91, -43.17), "RN": (-5.79, -35.20),
    "RO": (-8.76, -63.90), "RR": (2.82, -60.67), "RS": (-30.03, -51.23), "SC": (-27.59, -48.55),
    "SE": (-10.91, -37.07), "SP": (-23.55, -46.63), "TO": (-10.18, -48.33),
}

REGION_ORDER = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

NUMERIC_COLS_ITEMS = [
    "nota", "custo_medio_relatorio", "pis_cofins_recuperar", "icms_recuperar", "pis_cofins_recolher",
    "icms_recolher", "venda", "lucro_frete", "emb", "financeiro", "financeiro_pct",
    "mkt", "plat_log", "frete_item", "total_item", "qt", "mkm", "resultado_relatorio",
    "margem_relatorio_pct"
]

NUMERIC_COLS_PRICES = ["preco_custo", "preco_lista", "preco_promocao", "preco_desconto", "preco_ideal"]


def normalize_col(col: str) -> str:
    col = str(col).strip().lower()
    replacements = {
        "ç": "c", "ã": "a", "á": "a", "à": "a", "â": "a",
        "é": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o", "õ": "o", "ú": "u",
    }
    for old, new in replacements.items():
        col = col.replace(old, new)
    col = re.sub(r"[^a-z0-9]+", "_", col).strip("_")
    return col


def to_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    value = str(value).strip()
    if not value:
        return np.nan
    value = value.replace("R$", "").replace("%", "").strip()
    # pt-BR number
    if "," in value and "." in value:
        value = value.replace(".", "").replace(",", ".")
    elif "," in value:
        value = value.replace(",", ".")
    value = re.sub(r"[^0-9\.-]", "", value)
    if value in {"", ".", "-", "--"}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def brl(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v * 100:.2f}%".replace(".", ",")


def format_number_br(v: float, decimals: int = 2) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_int_br(v) -> str:
    if pd.isna(v):
        return "-"
    try:
        return f"{int(round(float(v))):,}".replace(",", ".")
    except Exception:
        return "-"


def format_currency_columns(df: pd.DataFrame, columns: list[str], prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{prefix}{format_number_br(x)}" if not pd.isna(x) else "-")
    return out


def format_percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].apply(pct)
    return out


def format_integer_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].apply(format_int_br)
    return out


def safe_div(a, b):
    if b in (0, 0.0) or pd.isna(b):
        return np.nan
    return a / b


def extract_uf(estado_destino: Optional[str], fallback_uf: Optional[str] = None) -> Optional[str]:
    if isinstance(estado_destino, str) and " - " in estado_destino:
        return estado_destino.split(" - ")[0].strip().upper()
    if isinstance(estado_destino, str) and len(estado_destino.strip()) == 2:
        return estado_destino.strip().upper()
    if isinstance(fallback_uf, str) and fallback_uf.strip():
        return fallback_uf.strip().upper()
    return None


def get_region(uf: Optional[str]) -> Optional[str]:
    if not uf:
        return None
    return STATE_TO_REGION.get(uf)


def find_default_sample(patterns: list[str]) -> Optional[Path]:
    base = Path("/mnt/data")
    for pattern in patterns:
        hits = sorted(base.glob(pattern))
        if hits:
            return hits[0]
    return None


# =========================
# Leitura / Parsing
# =========================

@st.cache_data(show_spinner=False)
def parse_price_workbook(file_bytes: bytes, file_name: str) -> tuple[pd.DataFrame, str]:
    bio = io.BytesIO(file_bytes)
    xl = pd.ExcelFile(bio)

    required_variants = [
        {"codigo", "preco_de_custo", "produto_derivacao"},
        {"codigo", "preco_de_custo", "produto"},
        {"codigo", "preco_de_custo"},
    ]

    best_sheet = None
    best_df = None

    for sheet_name in xl.sheet_names:
        try:
            temp = xl.parse(sheet_name)
        except Exception:
            continue

        normalized = {normalize_col(c): c for c in temp.columns}
        keys = set(normalized.keys())

        is_candidate = any(req.issubset(keys) for req in required_variants)
        if not is_candidate:
            continue

        temp = temp.rename(columns={v: k for k, v in normalized.items()})
        best_sheet = sheet_name
        best_df = temp.copy()
        if "2026" in sheet_name:
            break

    if best_df is None:
        raise ValueError(
            "Não foi encontrada uma aba com as colunas mínimas para custo/preço. "
            "Verifique se existe uma aba com CÓDIGO e Preço de custo."
        )

    rename_map = {
        "codigo": "codigo",
        "produto_derivacao": "produto_derivacao",
        "produto": "produto_derivacao",
        "preco_de_custo": "preco_custo",
        "preco": "preco_lista",
        "promocao": "preco_promocao",
        "preco_com_desconto": "preco_desconto",
        "preco_ideal": "preco_ideal",
        "fornecedor": "fornecedor",
    }
    cols = {c: rename_map[c] for c in best_df.columns if c in rename_map}
    df = best_df.rename(columns=cols).copy()

    for c in ["codigo", "produto_derivacao", "fornecedor", "preco_custo", "preco_lista", "preco_promocao", "preco_desconto", "preco_ideal"]:
        if c not in df.columns:
            df[c] = np.nan

    df["codigo"] = df["codigo"].apply(to_number).astype("Int64").astype(str)
    df = df[df["codigo"].ne("<NA>") & df["codigo"].ne("nan")].copy()

    for c in NUMERIC_COLS_PRICES:
        df[c] = df[c].apply(to_number)

    # Mantém o primeiro registro válido por código
    df = (
        df.sort_values(by=["preco_custo", "preco_desconto"], ascending=[False, False], na_position="last")
          .drop_duplicates(subset=["codigo"], keep="first")
    )
    return df.reset_index(drop=True), best_sheet


@st.cache_data(show_spinner=False)
def parse_margin_xls(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    raw = pd.read_excel(bio, sheet_name=0, header=None, engine="xlrd")

    rows = []
    current_code = None
    current_name = None

    for _, row in raw.iterrows():
        first = row.iloc[0]

        if isinstance(first, str) and first.strip().startswith("Produto:"):
            match = re.match(r"Produto:\s*(\d+)\s*-\s*(.*)", first.strip())
            if match:
                current_code = match.group(1).strip()
                current_name = match.group(2).strip()
            continue

        first_num = to_number(first)
        if not pd.isna(first_num):
            rows.append(
                {
                    "nota": int(first_num),
                    "produto_codigo": current_code,
                    "produto_nome_relatorio": current_name,
                    "deposito": row.iloc[1],
                    "recebimento": row.iloc[2],
                    "uf_relatorio": row.iloc[3],
                    "custo_medio_relatorio": row.iloc[4],
                    "pis_cofins_recuperar": row.iloc[5],
                    "icms_recuperar": row.iloc[6],
                    "pis_cofins_recolher": row.iloc[7],
                    "icms_recolher": row.iloc[8],
                    "venda": row.iloc[9],
                    "lucro_frete": row.iloc[10],
                    "emb": row.iloc[11],
                    "financeiro": row.iloc[12],
                    "financeiro_pct": row.iloc[13],
                    "mkt": row.iloc[14],
                    "plat_log": row.iloc[15],
                    "frete_item": row.iloc[16],
                    "total_item": row.iloc[17],
                    "qt": row.iloc[18],
                    "mkm": row.iloc[19],
                    "resultado_relatorio": row.iloc[20],
                    "margem_relatorio_pct": row.iloc[21],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Não foi possível extrair as linhas analíticas da planilha de margem bruta.")

    df["produto_codigo"] = df["produto_codigo"].astype(str)
    for c in NUMERIC_COLS_ITEMS:
        df[c] = df[c].apply(to_number)

    df["qt"] = df["qt"].fillna(0)
    return df


@st.cache_data(show_spinner=False)
def parse_consultar_notas_csv(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    df = pd.read_csv(bio, sep=";", encoding="latin1")
    df.columns = [normalize_col(c) for c in df.columns]

    rename_map = {
        "numero": "nota",
        "data_de_emissao": "data_emissao",
        "data_de_autorizacao": "data_autorizacao",
        "valor_total_frete": "frete_nf",
        "valor_total_produto": "valor_total_produto_nf",
        "valor_total_imposto": "valor_total_imposto_nf",
        "valor_desconto": "valor_desconto_nf",
        "valor_outras_despesas": "valor_outras_despesas_nf",
        "valor_total": "valor_total_nf",
        "transportadora": "transportadora",
        "servico_da_transportadora": "servico_transportadora",
        "estado_de_destino": "estado_destino",
        "pedido": "pedido",
        "nome_destinatario": "cliente",
        "nome_emissor": "emissor",
        "serie": "serie",
        "cfop": "cfop",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    required = ["nota", "frete_nf", "transportadora", "estado_destino"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"A coluna obrigatória '{c}' não foi encontrada no CSV de consultar notas.")

    df["nota"] = df["nota"].apply(to_number).astype("Int64")
    for c in ["frete_nf", "valor_total_produto_nf", "valor_total_imposto_nf", "valor_desconto_nf", "valor_outras_despesas_nf", "valor_total_nf"]:
        if c in df.columns:
            df[c] = df[c].apply(to_number)
        else:
            df[c] = np.nan

    for c in ["data_emissao", "data_autorizacao"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        else:
            df[c] = pd.NaT

    df["transportadora"] = df["transportadora"].fillna("Não informado")
    df["uf_nf"] = df.apply(lambda x: extract_uf(x.get("estado_destino"), None), axis=1)
    df["regiao"] = df["uf_nf"].apply(get_region)
    return df


@st.cache_data(show_spinner=False)
def build_model(price_bytes: bytes, price_name: str, margin_bytes: bytes, margin_name: str, notas_bytes: bytes, notas_name: str):
    prices, price_sheet = parse_price_workbook(price_bytes, price_name)
    items = parse_margin_xls(margin_bytes, margin_name)
    notas = parse_consultar_notas_csv(notas_bytes, notas_name)

    df = items.merge(
        prices,
        left_on="produto_codigo",
        right_on="codigo",
        how="left",
        validate="m:1",
    )

    notas_cols = [
        "nota", "data_emissao", "data_autorizacao", "transportadora", "servico_transportadora",
        "estado_destino", "uf_nf", "regiao", "frete_nf", "valor_total_produto_nf",
        "valor_total_imposto_nf", "valor_desconto_nf", "valor_total_nf", "pedido", "cliente"
    ]
    notas_subset = notas[[c for c in notas_cols if c in notas.columns]].copy()
    df = df.merge(notas_subset, on="nota", how="left", validate="m:1")

    # Campos padronizados
    df["produto_nome"] = df["produto_derivacao"].fillna(df["produto_nome_relatorio"])
    df["uf"] = df.apply(lambda x: extract_uf(x.get("estado_destino"), x.get("uf_relatorio")), axis=1)
    df["regiao"] = df["regiao"].fillna(df["uf"].apply(get_region))
    df["transportadora"] = df["transportadora"].fillna("Não informado")

    # Preços de referência
    df["preco_referencia_unit"] = np.where(
        df["preco_desconto"].fillna(0) > 0,
        df["preco_desconto"],
        np.where(df["preco_lista"].fillna(0) > 0, df["preco_lista"], np.nan)
    )

    # Cálculos principais sempre usando o custo da planilha de preços
    df["custo_unitario_tabela"] = df["preco_custo"]
    df["custo_total_tabela"] = df["custo_unitario_tabela"] * df["qt"]
    df["preco_venda_real_unit"] = df["venda"] / df["qt"].replace(0, np.nan)
    df["lucro_bruto_tabela"] = df["venda"] - df["custo_total_tabela"]
    df["margem_bruta_tabela_pct"] = df["lucro_bruto_tabela"] / df["venda"].replace(0, np.nan)

    df["impostos_reais"] = df[["pis_cofins_recolher", "icms_recolher"]].fillna(0).sum(axis=1)
    df["custos_operacionais"] = df[["emb", "financeiro", "mkt", "plat_log"]].fillna(0).sum(axis=1)
    df["lucro_liquido_estimado"] = (
        df["venda"].fillna(0)
        - df["custo_total_tabela"].fillna(0)
        - df["impostos_reais"].fillna(0)
        - df["custos_operacionais"].fillna(0)
        + df["lucro_frete"].fillna(0)
    )
    df["margem_liquida_estimada_pct"] = df["lucro_liquido_estimado"] / df["venda"].replace(0, np.nan)

    df["abaixo_do_custo"] = df["preco_venda_real_unit"] < df["custo_unitario_tabela"]
    df["abaixo_preco_referencia"] = df["preco_venda_real_unit"] < df["preco_referencia_unit"]
    df["tem_custo_cadastrado"] = df["custo_unitario_tabela"].notna()

    # Base de notas únicas para análises logísticas
    notas_unicas = notas.copy().drop_duplicates(subset=["nota"]).reset_index(drop=True)
    revenue_por_nota = df.groupby("nota", as_index=False).agg(
        receita_itens=("venda", "sum"),
        lucro_bruto_itens=("lucro_bruto_tabela", "sum"),
        lucro_liquido_itens=("lucro_liquido_estimado", "sum"),
        qtd_itens=("qt", "sum"),
    )
    notas_unicas = notas_unicas.merge(revenue_por_nota, on="nota", how="left")
    notas_unicas["frete_sobre_receita_pct"] = notas_unicas["frete_nf"] / notas_unicas["receita_itens"].replace(0, np.nan)

    diagnostics = {
        "price_sheet": price_sheet,
        "linhas_itens": len(df),
        "notas_unicas": int(df["nota"].nunique()),
        "produtos_unicos": int(df["produto_codigo"].nunique()),
        "custos_nao_encontrados": int(df["custo_unitario_tabela"].isna().sum()),
        "percentual_linhas_com_custo": float(df["tem_custo_cadastrado"].mean()) if len(df) else 0.0,
    }

    return df, notas_unicas, diagnostics


# =========================
# Visual helpers
# =========================

def apply_clean_layout(fig, height: int = 360):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title=None,
        separators=",.",
        hoverlabel=dict(namelength=-1),
    )
    return fig


def prepare_state_map(df_state: pd.DataFrame) -> pd.DataFrame:
    mapa = df_state.copy()
    mapa[["lat", "lon"]] = mapa["uf"].apply(lambda uf: pd.Series(UF_COORDS.get(uf, (np.nan, np.nan))))
    mapa = mapa.dropna(subset=["lat", "lon"])
    return mapa


def ordered_region_frame(df_region: pd.DataFrame, region_col: str = "regiao") -> pd.DataFrame:
    if region_col in df_region.columns:
        df_region[region_col] = pd.Categorical(df_region[region_col], categories=REGION_ORDER, ordered=True)
        df_region = df_region.sort_values(region_col)
    return df_region


def wrap_axis_label(value: str, width: int = 26) -> str:
    value = str(value or "").strip()
    if not value:
        return "Sem nome"
    words = value.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        add_len = len(word) + (1 if current else 0)
        if current and current_len + add_len > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines[:3])


# =========================
# Filtros e agregações
# =========================

def apply_filters(df: pd.DataFrame, notas: pd.DataFrame):
    st.sidebar.subheader("Filtros")

    min_date = df["data_emissao"].min()
    max_date = df["data_emissao"].max()

    period = None
    if pd.notna(min_date) and pd.notna(max_date):
        period = st.sidebar.date_input(
            "Período de emissão",
            value=(min_date.date(), max_date.date()),
            format="DD/MM/YYYY",
        )

    regioes = sorted([x for x in df["regiao"].dropna().unique().tolist()])
    transportadoras = sorted([x for x in df["transportadora"].dropna().unique().tolist()])
    ufs = sorted([x for x in df["uf"].dropna().unique().tolist()])

    regiao_sel = st.sidebar.multiselect("Região", regioes)
    uf_sel = st.sidebar.multiselect("UF", ufs)
    transp_sel = st.sidebar.multiselect("Transportadora", transportadoras)
    only_with_cost = st.sidebar.toggle("Somente itens com custo encontrado", value=True)

    df_f = df.copy()
    notas_f = notas.copy()

    if period and len(period) == 2:
        start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_f = df_f[df_f["data_emissao"].between(start, end, inclusive="both") | df_f["data_emissao"].isna()]
        notas_f = notas_f[notas_f["data_emissao"].between(start, end, inclusive="both") | notas_f["data_emissao"].isna()]

    if regiao_sel:
        df_f = df_f[df_f["regiao"].isin(regiao_sel)]
        notas_f = notas_f[notas_f["regiao"].isin(regiao_sel)]
    if uf_sel:
        df_f = df_f[df_f["uf"].isin(uf_sel)]
        notas_f = notas_f[notas_f["uf_nf"].isin(uf_sel)]
    if transp_sel:
        df_f = df_f[df_f["transportadora"].isin(transp_sel)]
        notas_f = notas_f[notas_f["transportadora"].isin(transp_sel)]
    if only_with_cost:
        df_f = df_f[df_f["tem_custo_cadastrado"]]

    return df_f, notas_f


# =========================
# Interface
# =========================

def metric_card(col, title: str, value: str, delta: Optional[str] = None):
    with col:
        st.metric(title, value, delta=delta)


def render_overview(df: pd.DataFrame, notas: pd.DataFrame):
    receita = df["venda"].sum()
    custo = df["custo_total_tabela"].sum()
    lucro_bruto = df["lucro_bruto_tabela"].sum()
    lucro_liq = df["lucro_liquido_estimado"].sum()
    frete_total_nf = notas["frete_nf"].sum()
    margem_bruta = safe_div(lucro_bruto, receita)
    margem_liq = safe_div(lucro_liq, receita)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metric_card(c1, "Receita de vendas", brl(receita))
    metric_card(c2, "Custo dos produtos", brl(custo))
    metric_card(c3, "Lucro bruto", brl(lucro_bruto), pct(margem_bruta) if not pd.isna(margem_bruta) else None)
    metric_card(c4, "Lucro líquido estimado", brl(lucro_liq), pct(margem_liq) if not pd.isna(margem_liq) else None)
    metric_card(c5, "Frete total NF", brl(frete_total_nf))
    metric_card(c6, "Notas fiscais", f"{int(df['nota'].nunique())}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Receita por região")
        por_regiao = (
            df.groupby("regiao", as_index=False)
              .agg(receita=("venda", "sum"))
        )
        por_regiao = ordered_region_frame(por_regiao)
        if not por_regiao.empty:
            fig = px.pie(por_regiao, names="regiao", values="receita", hole=0.45)
            st.plotly_chart(apply_clean_layout(fig, 360), use_container_width=True)
        else:
            st.info("Não há dados de região para exibir.")

    with right:
        st.markdown("### Receita, custo e lucro por região")
        desempenho = (
            df.groupby("regiao", as_index=False)
              .agg(
                  receita=("venda", "sum"),
                  custo=("custo_total_tabela", "sum"),
                  lucro_liquido=("lucro_liquido_estimado", "sum"),
              )
        )
        desempenho = ordered_region_frame(desempenho)
        if not desempenho.empty:
            melt = desempenho.melt(id_vars="regiao", value_vars=["receita", "custo", "lucro_liquido"], var_name="indicador", value_name="valor")
            fig = px.bar(melt, x="regiao", y="valor", color="indicador", barmode="group")
            st.plotly_chart(apply_clean_layout(fig, 360), use_container_width=True)
        else:
            st.info("Não há dados suficientes para comparar regiões.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Top 10 produtos por receita")
        top_revenue = (
            df.groupby(["produto_codigo", "produto_nome"], as_index=False)
              .agg(qt=("qt", "sum"), receita=("venda", "sum"))
              .sort_values("receita", ascending=False)
              .head(10)
              .sort_values("receita")
        )
        if not top_revenue.empty:
            fig = px.bar(top_revenue, x="receita", y="produto_nome", orientation="h", text_auto='.2s')
            st.plotly_chart(apply_clean_layout(fig, 420), use_container_width=True)
        else:
            st.info("Sem dados para ranking.")

    with c2:
        st.markdown("### Top 10 produtos mais vendidos")
        top_qt = (
            df.groupby(["produto_codigo", "produto_nome"], as_index=False)
              .agg(qt=("qt", "sum"), receita=("venda", "sum"))
              .sort_values("qt", ascending=False)
              .head(10)
              .sort_values("qt")
        )
        if not top_qt.empty:
            fig = px.bar(top_qt, x="qt", y="produto_nome", orientation="h", text_auto=True)
            st.plotly_chart(apply_clean_layout(fig, 420), use_container_width=True)
        else:
            st.info("Sem dados para ranking.")



def render_produtos(df: pd.DataFrame):
    st.markdown("### Produtos")

    resumo = (
        df.groupby(["produto_codigo", "produto_nome"], as_index=False)
          .agg(
              qt=("qt", "sum"),
              notas=("nota", "nunique"),
              receita=("venda", "sum"),
              custo=("custo_total_tabela", "sum"),
              lucro_bruto=("lucro_bruto_tabela", "sum"),
              lucro_liquido=("lucro_liquido_estimado", "sum"),
              frete_item=("frete_item", "sum"),
              preco_venda_real_unit=("preco_venda_real_unit", "mean"),
              custo_unitario_tabela=("custo_unitario_tabela", "mean"),
              preco_referencia_unit=("preco_referencia_unit", "mean"),
              abaixo_do_custo=("abaixo_do_custo", "sum"),
              abaixo_preco_referencia=("abaixo_preco_referencia", "sum"),
          )
    )
    resumo["margem_bruta_pct"] = resumo["lucro_bruto"] / resumo["receita"].replace(0, np.nan)
    resumo["margem_liquida_pct"] = resumo["lucro_liquido"] / resumo["receita"].replace(0, np.nan)
    resumo["gap_preco_unit"] = resumo["preco_venda_real_unit"] - resumo["custo_unitario_tabela"]
    resumo["produto_label"] = resumo["produto_nome"].apply(lambda x: wrap_axis_label(x, 28))
    resumo = resumo.sort_values("lucro_liquido", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Custo x preço de venda médio")
        comparativo = (
            resumo.dropna(subset=["custo_unitario_tabela", "preco_venda_real_unit"])
                  .sort_values("receita", ascending=False)
                  .head(10)
                  .sort_values("preco_venda_real_unit")
        )
        if not comparativo.empty:
            melt = comparativo.melt(
                id_vars=["produto_label"],
                value_vars=["custo_unitario_tabela", "preco_venda_real_unit"],
                var_name="tipo",
                value_name="valor",
            )
            melt["tipo"] = melt["tipo"].replace({
                "custo_unitario_tabela": "Custo unitário",
                "preco_venda_real_unit": "Preço médio de venda",
            })
            fig = px.bar(
                melt,
                x="valor",
                y="produto_label",
                color="tipo",
                orientation="h",
                barmode="group",
                text_auto='.2s',
            )
            st.plotly_chart(apply_clean_layout(fig, 500), use_container_width=True)
        else:
            st.info("Sem dados suficientes para o comparativo.")

    with c2:
        st.markdown("#### Produtos com menor margem líquida")
        low_margin = resumo.dropna(subset=["margem_liquida_pct"]).sort_values("margem_liquida_pct").head(10).sort_values("margem_liquida_pct")
        if not low_margin.empty:
            fig = px.bar(low_margin, x="margem_liquida_pct", y="produto_label", orientation="h", text_auto='.1%')
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(apply_clean_layout(fig, 500), use_container_width=True)
        else:
            st.info("Sem dados suficientes para análise de margem.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Produtos com maior diferença entre venda e custo")
        gap = resumo.dropna(subset=["gap_preco_unit"]).sort_values("gap_preco_unit", ascending=False).head(12).sort_values("gap_preco_unit")
        if not gap.empty:
            fig = px.bar(gap, x="gap_preco_unit", y="produto_label", orientation="h", text_auto='.2s')
            st.plotly_chart(apply_clean_layout(fig, 480), use_container_width=True)
        else:
            st.info("Sem dados suficientes para mostrar o gap de preço.")

    with c4:
        st.markdown("#### Top produtos por lucro líquido")
        top_profit = resumo.sort_values("lucro_liquido", ascending=False).head(12).sort_values("lucro_liquido")
        if not top_profit.empty:
            fig = px.bar(top_profit, x="lucro_liquido", y="produto_label", orientation="h", text_auto='.2s')
            st.plotly_chart(apply_clean_layout(fig, 480), use_container_width=True)
        else:
            st.info("Sem dados suficientes para mostrar o lucro por produto.")

    tabela = resumo.drop(columns=["produto_label"])
    tabela_exibicao = tabela.copy()
    tabela_exibicao = format_currency_columns(
        tabela_exibicao,
        [
            "receita", "custo", "lucro_bruto", "lucro_liquido", "frete_item",
            "preco_venda_real_unit", "custo_unitario_tabela", "preco_referencia_unit", "gap_preco_unit",
        ],
    )
    tabela_exibicao = format_percent_columns(tabela_exibicao, ["margem_bruta_pct", "margem_liquida_pct"])
    tabela_exibicao = format_integer_columns(tabela_exibicao, ["qt", "notas", "abaixo_do_custo", "abaixo_preco_referencia"])
    st.markdown("#### Tabela analítica por produto")
    st.dataframe(tabela_exibicao, use_container_width=True, hide_index=True)

    csv = tabela.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar análise de produtos (CSV)", csv, file_name="analise_produtos.csv", mime="text/csv")



def render_frete(notas: pd.DataFrame):
    st.markdown("### Frete e logística")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Resumo de frete por UF")
        por_uf = (
            notas.groupby("uf_nf", as_index=False)
                .agg(
                    frete_total=("frete_nf", "sum"),
                    frete_medio=("frete_nf", "mean"),
                    notas=("nota", "nunique"),
                    receita=("receita_itens", "sum"),
                )
                .rename(columns={"uf_nf": "UF"})
        )
        por_uf = por_uf[por_uf["UF"].isin(UF_COORDS.keys())].copy()
        if not por_uf.empty:
            por_uf["frete_sobre_receita_pct"] = por_uf["frete_total"] / por_uf["receita"].replace(0, np.nan)
            por_uf = por_uf.sort_values("frete_total", ascending=False)
            por_uf_exibicao = por_uf.copy()
            por_uf_exibicao = format_currency_columns(por_uf_exibicao, ["frete_total", "frete_medio", "receita"])
            por_uf_exibicao = format_percent_columns(por_uf_exibicao, ["frete_sobre_receita_pct"])
            por_uf_exibicao = format_integer_columns(por_uf_exibicao, ["notas"])
            st.dataframe(
                por_uf_exibicao,
                use_container_width=True,
                hide_index=True,
                height=460,
            )
        else:
            st.info("Sem dados de UF para montar a tabela.")

    with c2:
        st.markdown("#### Frete por região")
        por_regiao = notas.groupby("regiao", as_index=False).agg(frete=("frete_nf", "sum"), notas=("nota", "nunique")).dropna()
        por_regiao = ordered_region_frame(por_regiao)
        if not por_regiao.empty:
            fig = px.pie(por_regiao, names="regiao", values="frete", hole=0.45)
            st.plotly_chart(apply_clean_layout(fig, 460), use_container_width=True)
        else:
            st.info("Sem dados de região.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Frete total por transportadora")
        por_transp = (
            notas.groupby("transportadora", as_index=False)
                 .agg(frete=("frete_nf", "sum"), notas=("nota", "nunique"))
                 .sort_values("frete", ascending=False)
                 .head(12)
                 .sort_values("frete")
        )
        if not por_transp.empty:
            fig = px.bar(por_transp, x="frete", y="transportadora", orientation="h", text_auto='.2s')
            st.plotly_chart(apply_clean_layout(fig, 460), use_container_width=True)
        else:
            st.info("Sem dados de transportadora.")

    with c4:
        st.markdown("#### Frete médio por transportadora")
        frete_medio = (
            notas.groupby("transportadora", as_index=False)
                 .agg(
                     frete_medio=("frete_nf", "mean"),
                     frete_total=("frete_nf", "sum"),
                     notas=("nota", "nunique"),
                 )
                 .sort_values("frete_medio", ascending=False)
                 .head(12)
                 .sort_values("frete_medio")
        )
        if not frete_medio.empty:
            fig = px.bar(
                frete_medio,
                x="frete_medio",
                y="transportadora",
                orientation="h",
                text_auto='.2s',
                hover_data={"frete_total": ':.2f', "notas": True},
            )
            st.plotly_chart(apply_clean_layout(fig, 460), use_container_width=True)
        else:
            st.info("Sem dados suficientes para calcular o frete médio.")

    st.markdown("#### Frete por região e transportadora")
    cruzado = notas.groupby(["regiao", "transportadora"], as_index=False).agg(frete=("frete_nf", "sum")).dropna()
    cruzado = cruzado.sort_values("frete", ascending=False)
    top_transp = cruzado.groupby("transportadora")["frete"].sum().sort_values(ascending=False).head(8).index
    cruzado = cruzado[cruzado["transportadora"].isin(top_transp)]
    cruzado = ordered_region_frame(cruzado)
    if not cruzado.empty:
        fig = px.bar(cruzado, x="regiao", y="frete", color="transportadora", barmode="stack", text_auto='.2s')
        st.plotly_chart(apply_clean_layout(fig, 440), use_container_width=True)
    else:
        st.info("Sem dados suficientes para o empilhamento.")

    st.markdown("#### Tabela logística")
    tabela_logistica = notas[[c for c in ["nota", "data_emissao", "transportadora", "estado_destino", "regiao", "frete_nf", "receita_itens", "frete_sobre_receita_pct"] if c in notas.columns]].copy()
    if "data_emissao" in tabela_logistica.columns:
        tabela_logistica["data_emissao"] = pd.to_datetime(tabela_logistica["data_emissao"], errors="coerce").dt.strftime("%d/%m/%Y")
        tabela_logistica["data_emissao"] = tabela_logistica["data_emissao"].fillna("-")
    tabela_logistica = format_currency_columns(tabela_logistica, ["frete_nf", "receita_itens"])
    tabela_logistica = format_percent_columns(tabela_logistica, ["frete_sobre_receita_pct"])
    tabela_logistica = format_integer_columns(tabela_logistica, ["nota"])
    st.dataframe(
        tabela_logistica,
        use_container_width=True,
        hide_index=True,
    )



def render_analises_extras(df: pd.DataFrame, notas: pd.DataFrame):
    st.markdown("### Outras análises importantes")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Resultado por região")
        reg = (
            df.groupby("regiao", as_index=False)
              .agg(receita=("venda", "sum"), lucro_liquido=("lucro_liquido_estimado", "sum"), qt=("qt", "sum"))
        )
        reg["margem_liquida_pct"] = reg["lucro_liquido"] / reg["receita"].replace(0, np.nan)
        reg = ordered_region_frame(reg)
        if not reg.empty:
            fig = px.bar(reg, x="regiao", y="lucro_liquido", text_auto='.2s', hover_data={"receita": ':.2f', "qt": True, "margem_liquida_pct": ':.2%'})
            st.plotly_chart(apply_clean_layout(fig, 380), use_container_width=True)
        else:
            st.info("Sem dados suficientes.")

    with c2:
        st.markdown("#### Participação por meio de recebimento")
        rec = (
            df.groupby("recebimento", as_index=False)
              .agg(receita=("venda", "sum"), lucro_liquido=("lucro_liquido_estimado", "sum"), qt=("qt", "sum"))
              .sort_values("receita", ascending=False)
              .head(10)
        )
        if not rec.empty:
            fig = px.pie(rec, names="recebimento", values="receita", hole=0.45)
            st.plotly_chart(apply_clean_layout(fig, 380), use_container_width=True)
        else:
            st.info("Sem dados suficientes.")

    st.markdown("#### Regiões com maior peso de frete sobre a receita")
    peso_frete = notas.groupby("regiao", as_index=False).agg(frete=("frete_nf", "sum"), receita=("receita_itens", "sum")).dropna()
    peso_frete["frete_sobre_receita_pct"] = peso_frete["frete"] / peso_frete["receita"].replace(0, np.nan)
    peso_frete = ordered_region_frame(peso_frete)
    if not peso_frete.empty:
        fig = px.bar(peso_frete, x="regiao", y="frete_sobre_receita_pct", text_auto='.1%')
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(apply_clean_layout(fig, 360), use_container_width=True)
    else:
        st.info("Sem dados suficientes para medir o peso do frete.")

    st.markdown("#### Alertas de negócio")
    a1, a2, a3 = st.columns(3)
    abaixo_custo = int(df["abaixo_do_custo"].sum())
    abaixo_referencia = int(df["abaixo_preco_referencia"].sum())
    frete_pesado = int((notas["frete_sobre_receita_pct"] > 0.15).fillna(False).sum())
    a1.error(f"Itens vendidos abaixo do custo: {abaixo_custo}")
    a2.warning(f"Itens vendidos abaixo do preço de referência: {abaixo_referencia}")
    a3.info(f"Notas com frete acima de 15% da receita: {frete_pesado}")

    st.markdown("#### Itens com custo não encontrado na planilha de preços")
    missing = df[df["custo_unitario_tabela"].isna()][["nota", "produto_codigo", "produto_nome", "venda", "qt", "transportadora", "estado_destino"]].drop_duplicates()
    if not missing.empty:
        missing_exibicao = missing.copy()
        missing_exibicao = format_currency_columns(missing_exibicao, ["venda"])
        missing_exibicao = format_integer_columns(missing_exibicao, ["nota", "qt"])
        st.dataframe(missing_exibicao, use_container_width=True, hide_index=True)
    else:
        st.success("Todos os itens filtrados possuem custo encontrado na planilha de preços.")



def render_diagnostics(diag: dict):
    with st.expander("Ver diagnóstico da carga"):
        col1, col2, col3 = st.columns(3)
        col1.write(f"**Aba de preços utilizada:** {diag['price_sheet']}")
        col1.write(f"**Linhas analíticas:** {diag['linhas_itens']}")
        col2.write(f"**Notas únicas:** {diag['notas_unicas']}")
        col2.write(f"**Produtos únicos:** {diag['produtos_unicos']}")
        col3.write(f"**Custos não encontrados:** {diag['custos_nao_encontrados']}")
        col3.write(f"**Cobertura de custo:** {pct(diag['percentual_linhas_com_custo'])}")


# =========================
# App
# =========================


def main():
    st.title("📊 Dashboard de Margem BI em Streamlit")
    st.caption(
        "Cruza as 3 planilhas para analisar margem por produto, custo x venda, frete por transportadora/região, "
        "produtos mais vendidos e indicadores operacionais com visual simples em barras, pizza, tabelas e empilhados."
    )

    with st.sidebar:
        st.markdown("### Arquivos de entrada")
        st.write("Envie os 3 arquivos para atualizar o dashboard.")

    sample_price = find_default_sample(["*Planilha Geral de Preços*.xlsx", "*.xlsx"])
    sample_margin = find_default_sample(["*notasFiscaisFaturadas*.xls", "*.xls"])
    sample_notas = find_default_sample(["*Consultar Notas Fiscais*.csv", "*.csv"])

    price_file = st.sidebar.file_uploader("1) Planilha de preços/custo", type=["xlsx"])
    margin_file = st.sidebar.file_uploader("2) Relatório de margem bruta das NF faturadas", type=["xls"])
    notas_file = st.sidebar.file_uploader("3) CSV consultar notas fiscais", type=["csv"])

    use_samples = False
    if sample_price and sample_margin and sample_notas:
        use_samples = st.sidebar.toggle("Usar arquivos de exemplo já carregados", value=not (price_file or margin_file or notas_file))

    if use_samples:
        price_name = sample_price.name
        margin_name = sample_margin.name
        notas_name = sample_notas.name
        price_bytes = sample_price.read_bytes()
        margin_bytes = sample_margin.read_bytes()
        notas_bytes = sample_notas.read_bytes()
    else:
        if not (price_file and margin_file and notas_file):
            st.info("Envie os 3 arquivos ou ative a opção de usar os arquivos de exemplo já carregados.")
            st.stop()
        price_name = price_file.name
        margin_name = margin_file.name
        notas_name = notas_file.name
        price_bytes = price_file.getvalue()
        margin_bytes = margin_file.getvalue()
        notas_bytes = notas_file.getvalue()

    try:
        df, notas, diagnostics = build_model(price_bytes, price_name, margin_bytes, margin_name, notas_bytes, notas_name)
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.stop()

    render_diagnostics(diagnostics)
    df_f, notas_f = apply_filters(df, notas)

    tab1, tab2, tab3, tab4 = st.tabs(["Visão geral", "Produtos", "Frete & logística", "Análises extras"])

    with tab1:
        render_overview(df_f, notas_f)
    with tab2:
        render_produtos(df_f)
    with tab3:
        render_frete(notas_f)
    with tab4:
        render_analises_extras(df_f, notas_f)


if __name__ == "__main__":
    main()
