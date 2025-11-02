"""
Página de seleção de ativos - Versão Simplificada
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Selecionar Ativos",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# INICIALIZAÇÃO GARANTIDA
# ==========================================
def init_state():
    """Inicializa session state"""
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = []

# CHAMAR ANTES DE TUDO
init_state()

# ==========================================
# DADOS PADRÃO
# ==========================================
def get_ativos():
    """Retorna lista de ativos"""
    return pd.DataFrame({
        'ticker': [
            'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3',
            'WEGE3', 'RENT3', 'LREN3', 'MGLU3', 'B3SA3',
            'BBAS3', 'SUZB3', 'RAIL3', 'JBSS3', 'EMBR3',
            'HGLG11', 'MXRF11', 'KNRI11', 'XPML11', 'VISC11',
            'BOVA11', 'SMAL11', 'IVVB11'
        ],
        'nome': [
            'Petrobras', 'Vale', 'Itaú', 'Bradesco', 'Ambev',
            'Weg', 'Localiza', 'Lojas Renner', 'Magazine Luiza', 'B3',
            'Banco do Brasil', 'Suzano', 'Rumo', 'JBS', 'Embraer',
            'CSHG Logística', 'Maxi Renda', 'Kinea Renda', 'XP Malls', 'Vinci Shopping',
            'Ibovespa', 'Small Caps', 'S&P 500'
        ],
        'tipo': [
            'ACAO', 'ACAO', 'ACAO', 'ACAO', 'ACAO',
            'ACAO', 'ACAO', 'ACAO', 'ACAO', 'ACAO',
            'ACAO', 'ACAO', 'ACAO', 'ACAO', 'ACAO',
            'FII', 'FII', 'FII', 'FII', 'FII',
            'ETF', 'ETF', 'ETF'
        ]
    })

# ==========================================
# INTERFACE
# ==========================================

st.title("📊 Seleção de Ativos")
st.markdown("---")

# Obter dados
df = get_ativos()

# Filtros
col1, col2 = st.columns(2)

with col1:
    tipo_filtro = st.selectbox(
        "Tipo",
        ['TODOS', 'ACAO', 'FII', 'ETF']
    )

with col2:
    busca = st.text_input("Buscar", placeholder="Digite código ou nome...")

# Aplicar filtros
df_filtrado = df.copy()

if tipo_filtro != 'TODOS':
    df_filtrado = df_filtrado[df_filtrado['tipo'] == tipo_filtro]

if busca:
    busca = busca.upper()
    df_filtrado = df_filtrado[
        df_filtrado['ticker'].str.contains(busca) |
        df_filtrado['nome'].str.upper().str.contains(busca)
    ]

# Métricas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Disponíveis", len(df))
with col2:
    st.metric("Filtrados", len(df_filtrado))
with col3:
    st.metric("Selecionados", len(st.session_state.selected_tickers))

st.markdown("---")

# Tabela com seleção
if not df_filtrado.empty:
    df_display = df_filtrado.copy()
    df_display['Selecionar'] = df_display['ticker'].isin(
        st.session_state.selected_tickers
    )
    
    edited = st.data_editor(
        df_display,
        column_config={
            "Selecionar": st.column_config.CheckboxColumn(
                "Selecionar",
                default=False
            )
        },
        disabled=["ticker", "nome", "tipo"],
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Atualizar seleção
    st.session_state.selected_tickers = edited[
        edited['Selecionar']
    ]['ticker'].tolist()

else:
    st.warning("Nenhum ativo encontrado")

st.markdown("---")

# Ativos selecionados
st.header("✅ Selecionados")

if st.session_state.selected_tickers:
    df_sel = df[df['ticker'].isin(st.session_state.selected_tickers)]
    
    st.dataframe(
        df_sel,
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Salvar", type="primary", use_container_width=True):
            st.session_state.portfolio_tickers = st.session_state.selected_tickers.copy()
            st.success(f"✓ {len(st.session_state.portfolio_tickers)} ativos salvos!")
            st.balloons()
    
    with col2:
        if st.button("🗑️ Limpar", use_container_width=True):
            st.session_state.selected_tickers = []
            st.rerun()
else:
    st.info("Nenhum ativo selecionado")
