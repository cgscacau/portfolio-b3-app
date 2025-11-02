"""
Página de seleção de ativos
Permite selecionar ações, FIIs e outros ativos da B3
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Configuração da página
st.set_page_config(
    page_title="Selecionar Ativos",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# INICIALIZAÇÃO NO NÍVEL DO MÓDULO
# ==========================================
if 'universe_df' not in st.session_state:
    st.session_state.universe_df = pd.DataFrame()

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = []


def carregar_universo_b3():
    """
    Carrega o universo de ativos da B3
    
    Returns:
        DataFrame com ativos disponíveis
    """
    try:
        # Tentar carregar arquivo CSV
        csv_path = root_dir / 'assets' / 'b3_universe.csv'
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            st.success(f"✓ {len(df)} ativos carregados do arquivo")
            return df
        else:
            st.warning("⚠ Arquivo b3_universe.csv não encontrado")
            return criar_universo_padrao()
            
    except Exception as e:
        st.error(f"Erro ao carregar universo: {str(e)}")
        return criar_universo_padrao()


def criar_universo_padrao():
    """
    Cria um universo padrão de ativos caso o arquivo não exista
    
    Returns:
        DataFrame com ativos padrão
    """
    ativos_padrao = {
        'ticker': [
            # Ações principais
            'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3',
            'WEGE3', 'RENT3', 'LREN3', 'MGLU3', 'B3SA3',
            'BBAS3', 'SUZB3', 'RAIL3', 'JBSS3', 'EMBR3',
            'RADL3', 'VIVT3', 'GGBR4', 'CSNA3', 'USIM5',
            # FIIs principais
            'HGLG11', 'MXRF11', 'KNRI11', 'XPML11', 'VISC11',
            'BTLG11', 'HGRU11', 'KNCR11', 'PVBI11', 'IRDM11',
            # ETFs
            'BOVA11', 'SMAL11', 'IVVB11', 'PIBB11'
        ],
        'nome': [
            'Petrobras', 'Vale', 'Itaú', 'Bradesco', 'Ambev',
            'Weg', 'Localiza', 'Lojas Renner', 'Magazine Luiza', 'B3',
            'Banco do Brasil', 'Suzano', 'Rumo', 'JBS', 'Embraer',
            'Raia Drogasil', 'Vivo', 'Gerdau', 'CSN', 'Usiminas',
            'CSHG Logística', 'Maxi Renda', 'Kinea Renda', 'XP Malls', 'Vinci Shopping',
            'BTG Logística', 'CSHG Renda Urbana', 'Kinea Crédito', 'PV Birigui', 'Iridium',
            'Ibovespa', 'Small Caps', 'S&P 500', 'IBrX'
        ],
        'tipo': [
            'ACAO', 'ACAO', 'ACAO', 'ACAO', 'ACAO',
            'ACAO', 'ACAO', 'ACAO', 'ACAO', 'ACAO',
            'ACAO', 'ACAO', 'ACAO', 'ACAO', 'ACAO',
            'ACAO', 'ACAO', 'ACAO', 'ACAO', 'ACAO',
            'FII', 'FII', 'FII', 'FII', 'FII',
            'FII', 'FII', 'FII', 'FII', 'FII',
            'ETF', 'ETF', 'ETF', 'ETF'
        ]
    }
    
    df = pd.DataFrame(ativos_padrao)
    st.info(f"ℹ Usando {len(df)} ativos padrão")
    
    return df


def filtrar_ativos(df, tipo_filtro, busca_texto):
    """
    Filtra DataFrame de ativos
    
    Args:
        df: DataFrame com ativos
        tipo_filtro: Tipo de ativo (TODOS, ACAO, FII, ETF)
        busca_texto: Texto para buscar
        
    Returns:
        DataFrame filtrado
    """
    df_filtrado = df.copy()
    
    # Filtrar por tipo
    if tipo_filtro != 'TODOS':
        df_filtrado = df_filtrado[df_filtrado['tipo'] == tipo_filtro]
    
    # Filtrar por texto
    if busca_texto:
        busca_texto = busca_texto.upper()
        mask = (
            df_filtrado['ticker'].str.contains(busca_texto, na=False) |
            df_filtrado['nome'].str.contains(busca_texto, case=False, na=False)
        )
        df_filtrado = df_filtrado[mask]
    
    return df_filtrado


def exibir_seletor_ativos(df):
    """
    Exibe interface de seleção de ativos
    
    Args:
        df: DataFrame com ativos disponíveis
    """
    st.header("📊 Selecionar Ativos")
    
    # Filtros
    col1, col2 = st.columns([1, 2])
    
    with col1:
        tipo_filtro = st.selectbox(
            "Tipo de Ativo",
            options=['TODOS', 'ACAO', 'FII', 'ETF'],
            index=0
        )
    
    with col2:
        busca_texto = st.text_input(
            "Buscar por código ou nome",
            placeholder="Ex: PETR, Petrobras, HGLG..."
        )
    
    # Aplicar filtros
    df_filtrado = filtrar_ativos(df, tipo_filtro, busca_texto)
    
    # Estatísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Disponível", len(df))
    with col2:
        st.metric("Filtrados", len(df_filtrado))
    with col3:
        st.metric("Selecionados", len(st.session_state.selected_tickers))
    
    st.markdown("---")
    
    # Tabela de seleção
    if not df_filtrado.empty:
        # Adicionar coluna de seleção
        df_display = df_filtrado.copy()
        df_display['Selecionar'] = df_display['ticker'].isin(st.session_state.selected_tickers)
        
        # Configurar editor
        edited_df = st.data_editor(
            df_display,
            column_config={
                "Selecionar": st.column_config.CheckboxColumn(
                    "Selecionar",
                    help="Marque para adicionar ao portfólio",
                    default=False,
                ),
                "ticker": st.column_config.TextColumn(
                    "Código",
                    width="small",
                ),
                "nome": st.column_config.TextColumn(
                    "Nome",
                    width="large",
                ),
                "tipo": st.column_config.TextColumn(
                    "Tipo",
                    width="small",
                ),
            },
            disabled=["ticker", "nome", "tipo"],
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # Atualizar seleção
        selecionados = edited_df[edited_df['Selecionar']]['ticker'].tolist()
        st.session_state.selected_tickers = selecionados
        
    else:
        st.warning("⚠ Nenhum ativo encontrado com os filtros aplicados")


def exibir_ativos_selecionados():
    """Exibe lista de ativos selecionados"""
    st.header("✅ Ativos Selecionados")
    
    if st.session_state.selected_tickers:
        # Criar DataFrame com selecionados
        df_selecionados = st.session_state.universe_df[
            st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
        ].copy()
        
        # Contar por tipo
        contagem = df_selecionados['tipo'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(df_selecionados))
        with col2:
            st.metric("Ações", contagem.get('ACAO', 0))
        with col3:
            st.metric("FIIs", contagem.get('FII', 0))
        with col4:
            st.metric("ETFs", contagem.get('ETF', 0))
        
        # Exibir tabela
        st.dataframe(
            df_selecionados[['ticker', 'nome', 'tipo']],
            use_container_width=True,
            hide_index=True
        )
        
        # Botões de ação
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Salvar Seleção", type="primary", use_container_width=True):
                st.session_state.portfolio_tickers = st.session_state.selected_tickers.copy()
                st.success(f"✓ {len(st.session_state.portfolio_tickers)} ativos salvos no portfólio!")
                st.balloons()
        
        with col2:
            if st.button("🗑️ Limpar Seleção", use_container_width=True):
                st.session_state.selected_tickers = []
                st.rerun()
        
    else:
        st.info("ℹ Nenhum ativo selecionado ainda. Use a tabela acima para selecionar.")


def main():
    """Função principal"""
    
    # Título
    st.title("📊 Seleção de Ativos")
    st.markdown("Selecione os ativos que deseja acompanhar no seu portfólio.")
    st.markdown("---")
    
    # Carregar universo se vazio (session_state já foi inicializado no topo)
    if st.session_state.universe_df.empty:
        with st.spinner("Carregando universo de ativos..."):
            st.session_state.universe_df = carregar_universo_b3()
    
    # Verificar se carregou
    if st.session_state.universe_df.empty:
        st.error("❌ Não foi possível carregar os ativos.")
        st.stop()
    
    # Exibir seletor
    exibir_seletor_ativos(st.session_state.universe_df)
    
    st.markdown("---")
    
    # Exibir selecionados
    exibir_ativos_selecionados()
    
    # Informações adicionais
    with st.expander("ℹ️ Informações"):
        st.markdown("""
        **Como usar:**
        1. Use os filtros para encontrar ativos
        2. Marque a caixa "Selecionar" dos ativos desejados
        3. Clique em "Salvar Seleção" para confirmar
        4. Os ativos salvos estarão disponíveis nas outras páginas
        
        **Tipos de ativos:**
        - **ACAO**: Ações de empresas listadas na B3
        - **FII**: Fundos Imobiliários
        - **ETF**: Fundos de Índice (Exchange Traded Funds)
        """)


if __name__ == "__main__":
    main()
