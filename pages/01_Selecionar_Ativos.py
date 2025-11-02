"""
📊 Seleção de Ativos
Lista de ativos negociados nos últimos 30 dias com filtros avançados
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Configurar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data
from core.init import init_all

# Configuração da página
st.set_page_config(
    page_title="Selecionar Ativos",
    page_icon="📊",
    layout="wide"
)

# Inicializar
init_all()


# ==========================================
# CARREGAR UNIVERSO B3
# ==========================================

@st.cache_data(ttl=86400, show_spinner=False)
def carregar_universo_b3():
    """
    Carrega universo de ativos do arquivo CSV ou cria padrão
    
    Returns:
        DataFrame com [ticker, nome, setor, segmento, tipo]
    """
    csv_path = root_dir / 'assets' / 'b3_universe.csv'
    
    # Tentar carregar do CSV
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            
            # Validar colunas necessárias
            required_cols = ['ticker', 'nome', 'setor', 'segmento', 'tipo']
            if all(col in df.columns for col in required_cols):
                # Limpar dados
                df['ticker'] = df['ticker'].str.upper().str.strip()
                df = df.dropna(subset=['ticker'])
                df = df.drop_duplicates(subset=['ticker'])
                
                return df
        except Exception as e:
            st.warning(f"⚠️ Erro ao ler CSV: {str(e)}. Usando universo padrão.")
    
    # Se não conseguiu carregar, criar universo padrão
    return criar_universo_padrao()


def criar_universo_padrao():
    """
    Cria universo padrão de ativos caso CSV não exista
    
    Returns:
        DataFrame com ativos padrão
    """
    ativos = []
    
    # FINANCEIRO
    ativos.extend([
        ('ITUB4', 'Itaú Unibanco', 'Financeiro', 'Bancos', 'ACAO'),
        ('BBDC4', 'Bradesco', 'Financeiro', 'Bancos', 'ACAO'),
        ('BBAS3', 'Banco do Brasil', 'Financeiro', 'Bancos', 'ACAO'),
        ('SANB11', 'Santander', 'Financeiro', 'Bancos', 'ACAO'),
        ('BBSE3', 'BB Seguridade', 'Financeiro', 'Seguros', 'ACAO'),
        ('B3SA3', 'B3', 'Financeiro', 'Serviços Financeiros', 'ACAO'),
    ])
    
    # ENERGIA
    ativos.extend([
        ('PETR4', 'Petrobras PN', 'Energia', 'Petróleo e Gás', 'ACAO'),
        ('PETR3', 'Petrobras ON', 'Energia', 'Petróleo e Gás', 'ACAO'),
        ('PRIO3', 'PetroRio', 'Energia', 'Petróleo e Gás', 'ACAO'),
        ('ELET3', 'Eletrobras', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('ELET6', 'Eletrobras PNB', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('ENBR3', 'Energias BR', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('CPFE3', 'CPFL Energia', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('TAEE11', 'Taesa', 'Energia', 'Energia Elétrica', 'ACAO'),
    ])
    
    # MATERIAIS BÁSICOS
    ativos.extend([
        ('VALE3', 'Vale', 'Materiais Básicos', 'Mineração', 'ACAO'),
        ('CSNA3', 'CSN', 'Materiais Básicos', 'Siderurgia', 'ACAO'),
        ('GGBR4', 'Gerdau', 'Materiais Básicos', 'Siderurgia', 'ACAO'),
        ('USIM5', 'Usiminas', 'Materiais Básicos', 'Siderurgia', 'ACAO'),
        ('SUZB3', 'Suzano', 'Materiais Básicos', 'Papel e Celulose', 'ACAO'),
    ])
    
    # CONSUMO
    ativos.extend([
        ('ABEV3', 'Ambev', 'Consumo', 'Bebidas', 'ACAO'),
        ('JBSS3', 'JBS', 'Consumo', 'Alimentos', 'ACAO'),
        ('LREN3', 'Lojas Renner', 'Consumo', 'Varejo', 'ACAO'),
        ('MGLU3', 'Magazine Luiza', 'Consumo', 'Varejo', 'ACAO'),
        ('CRFB3', 'Carrefour Brasil', 'Consumo', 'Varejo', 'ACAO'),
        ('ASAI3', 'Assaí', 'Consumo', 'Varejo', 'ACAO'),
    ])
    
    # SAÚDE
    ativos.extend([
        ('RADL3', 'Raia Drogasil', 'Saúde', 'Farmácias', 'ACAO'),
        ('FLRY3', 'Fleury', 'Saúde', 'Serviços Médicos', 'ACAO'),
        ('HAPV3', 'Hapvida', 'Saúde', 'Saúde', 'ACAO'),
    ])
    
    # INDUSTRIAL
    ativos.extend([
        ('WEGE3', 'WEG', 'Industrial', 'Máquinas e Equipamentos', 'ACAO'),
        ('EMBR3', 'Embraer', 'Industrial', 'Aeronáutica', 'ACAO'),
        ('RAIL3', 'Rumo', 'Industrial', 'Transporte', 'ACAO'),
        ('CCRO3', 'CCR', 'Industrial', 'Concessões', 'ACAO'),
    ])
    
    # TECNOLOGIA
    ativos.extend([
        ('VIVT3', 'Vivo', 'Tecnologia', 'Telecomunicações', 'ACAO'),
        ('TIMS3', 'Tim', 'Tecnologia', 'Telecomunicações', 'ACAO'),
        ('TOTS3', 'Totvs', 'Tecnologia', 'Software', 'ACAO'),
    ])
    
    # IMOBILIÁRIO
    ativos.extend([
        ('CYRE3', 'Cyrela', 'Imobiliário', 'Construção', 'ACAO'),
        ('MRVE3', 'MRV', 'Imobiliário', 'Construção', 'ACAO'),
        ('RENT3', 'Localiza', 'Imobiliário', 'Aluguel de Veículos', 'ACAO'),
    ])
    
    # UTILIDADES
    ativos.extend([
        ('SBSP3', 'Sabesp', 'Utilidades', 'Água e Saneamento', 'ACAO'),
    ])
    
    # EDUCAÇÃO
    ativos.extend([
        ('YDUQ3', 'Yduqs', 'Educação', 'Educação', 'ACAO'),
        ('COGN3', 'Cogna', 'Educação', 'Educação', 'ACAO'),
    ])
    
    # FIIs
    ativos.extend([
        ('HGLG11', 'CSHG Logística', 'Fundos Imobiliários', 'Logística', 'FII'),
        ('MXRF11', 'Maxi Renda', 'Fundos Imobiliários', 'Lajes Corporativas', 'FII'),
        ('KNRI11', 'Kinea Renda', 'Fundos Imobiliários', 'Lajes Corporativas', 'FII'),
        ('XPML11', 'XP Malls', 'Fundos Imobiliários', 'Shopping', 'FII'),
        ('VISC11', 'Vinci Shopping', 'Fundos Imobiliários', 'Shopping', 'FII'),
        ('BTLG11', 'BTG Logística', 'Fundos Imobiliários', 'Logística', 'FII'),
        ('HGRU11', 'CSHG Renda Urbana', 'Fundos Imobiliários', 'Multiestratégia', 'FII'),
        ('KNCR11', 'Kinea Crédito', 'Fundos Imobiliários', 'Crédito', 'FII'),
        ('PVBI11', 'PV Birigui', 'Fundos Imobiliários', 'Lajes Corporativas', 'FII'),
        ('IRDM11', 'Iridium', 'Fundos Imobiliários', 'Lajes Corporativas', 'FII'),
    ])
    
    # ETFs
    ativos.extend([
        ('BOVA11', 'Ibovespa', 'ETFs', 'Índice', 'ETF'),
        ('SMAL11', 'Small Caps', 'ETFs', 'Índice', 'ETF'),
        ('IVVB11', 'S&P 500', 'ETFs', 'Índice', 'ETF'),
        ('PIBB11', 'IBrX', 'ETFs', 'Índice', 'ETF'),
    ])
    
    # Criar DataFrame
    df = pd.DataFrame(ativos, columns=['ticker', 'nome', 'setor', 'segmento', 'tipo'])
    
    return df


# ==========================================
# FILTRO DE LIQUIDEZ (30 DIAS)
# ==========================================

def verificar_liquidez_ativos(tickers, min_sessoes=5):
    """
    Verifica quais ativos foram negociados nos últimos 30 dias
    
    Args:
        tickers: Lista de tickers
        min_sessoes: Mínimo de sessões com dados
        
    Returns:
        Dict {ticker: {'negociado': bool, 'sessoes': int}}
    """
    resultado = {}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=35)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Verificando {ticker} ({idx+1}/{len(tickers)})...")
        
        try:
            hist = data.get_price_history([ticker], start_date, end_date, use_cache=True)
            
            if not hist.empty and ticker in hist.columns:
                dados_validos = hist[ticker].dropna()
                sessoes = len(dados_validos)
                negociado = sessoes >= min_sessoes
            else:
                sessoes = 0
                negociado = False
            
            resultado[ticker] = {
                'negociado': negociado,
                'sessoes': sessoes
            }
            
        except:
            resultado[ticker] = {
                'negociado': False,
                'sessoes': 0
            }
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    return resultado


# ==========================================
# SELEÇÃO INTELIGENTE
# ==========================================

def selecionar_por_criterio(df, criterio='liquidez', n=10):
    """
    Seleciona ativos por critério específico
    
    Args:
        df: DataFrame com ativos
        criterio: 'liquidez', 'dy', 'setor'
        n: Número de ativos
        
    Returns:
        Lista de tickers
    """
    if criterio == 'liquidez':
        # Ativos mais líquidos (blue chips conhecidos)
        blue_chips = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 'ABEV3', 
                      'WEGE3', 'B3SA3', 'RENT3', 'ELET3']
        return [t for t in blue_chips if t in df['ticker'].values][:n]
    
    elif criterio == 'dy':
        # Ativos conhecidos por bons dividendos
        alto_dy = ['ITUB4', 'BBDC4', 'BBAS3', 'PETR4', 'VALE3', 'TAEE11',
                   'HGLG11', 'MXRF11', 'KNRI11', 'XPML11']
        return [t for t in alto_dy if t in df['ticker'].values][:n]
    
    elif criterio == 'fiis':
        # Apenas FIIs
        df_fiis = df[df['tipo'] == 'FII']
        return df_fiis['ticker'].head(n).tolist()
    
    return []


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal da página"""
    
    st.title("📊 Seleção de Ativos")
    st.markdown("Selecione ativos da B3 para análise de portfólio e dividendos")
    st.markdown("---")
    
    # Carregar universo
    with st.spinner("📥 Carregando universo B3..."):
        df_universo = carregar_universo_b3()
    
    # Verificar se carregou do CSV ou padrão
    csv_path = root_dir / 'assets' / 'b3_universe.csv'
    if csv_path.exists():
        st.success(f"✅ **{len(df_universo)} ativos** carregados do arquivo CSV")
    else:
        st.info(f"ℹ️ **{len(df_universo)} ativos** no universo padrão (crie `assets/b3_universe.csv` para personalizar)")
    
    # Inicializar estado
    if 'df_filtrado' not in st.session_state:
        st.session_state.df_filtrado = df_universo.copy()
        st.session_state.df_filtrado['verificado_30d'] = False
        st.session_state.df_filtrado['sessoes_ativas'] = 0
    
    # Sidebar - Filtros
    with st.sidebar:
        st.header("🔍 Filtros")
        
        # Tipo de ativo
        st.subheader("📋 Tipo")
        tipos_disponiveis = sorted(df_universo['tipo'].unique())
        tipos_selecionados = st.multiselect(
            "Selecione os tipos",
            options=tipos_disponiveis,
            default=tipos_disponiveis,
            help="Ações, FIIs ou ETFs"
        )
        
        st.markdown("---")
        
        # Setor
        st.subheader("🏢 Setor")
        setores_disponiveis = ['Todos'] + sorted(df_universo['setor'].unique())
        setor_selecionado = st.selectbox(
            "Filtrar por setor",
            options=setores_disponiveis,
            help="Escolha um setor específico"
        )
        
        st.markdown("---")
        
        # Segmento
        st.subheader("🎯 Segmento")
        
        # Filtrar segmentos baseado no setor
        if setor_selecionado != 'Todos':
            segmentos_disponiveis = ['Todos'] + sorted(
                df_universo[df_universo['setor'] == setor_selecionado]['segmento'].unique()
            )
        else:
            segmentos_disponiveis = ['Todos'] + sorted(df_universo['segmento'].unique())
        
        segmento_selecionado = st.selectbox(
            "Filtrar por segmento",
            options=segmentos_disponiveis,
            help="Escolha um segmento específico"
        )
        
        st.markdown("---")
        
        # Busca por texto
        st.subheader("🔎 Busca")
        texto_busca = st.text_input(
            "Ticker ou Nome",
            placeholder="Ex: PETR4, Petrobras",
            help="Digite parte do código ou nome"
        )
        
        st.markdown("---")
        
        # Verificação de liquidez
        st.subheader("📈 Liquidez (30 dias)")
        
        verificar_liquidez = st.checkbox(
            "Verificar negociação",
            value=False,
            help="Verifica quais ativos foram negociados nos últimos 30 dias (pode demorar)"
        )
        
        if verificar_liquidez:
            min_sessoes = st.slider(
                "Mínimo de sessões",
                min_value=1,
                max_value=20,
                value=5,
                help="Dias mínimos com negociação"
            )
            
            apenas_negociados = st.checkbox(
                "Apenas negociados",
                value=True,
                help="Mostrar apenas ativos que passaram no filtro"
            )
        else:
            min_sessoes = 5
            apenas_negociados = False
        
        st.markdown("---")
        
        # Botão aplicar
        btn_aplicar = st.button(
            "🔄 Aplicar Filtros",
            type="primary",
            use_container_width=True
        )
    
    # Aplicar filtros quando botão clicado
    if btn_aplicar:
        
        with st.spinner("🔍 Aplicando filtros..."):
            df_filtrado = df_universo.copy()
            
            # Filtro por tipo
            if tipos_selecionados:
                df_filtrado = df_filtrado[df_filtrado['tipo'].isin(tipos_selecionados)]
            
            # Filtro por setor
            if setor_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['setor'] == setor_selecionado]
            
            # Filtro por segmento
            if segmento_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['segmento'] == segmento_selecionado]
            
            # Busca por texto
            if texto_busca:
                texto = texto_busca.upper()
                mask = (
                    df_filtrado['ticker'].str.contains(texto, na=False) |
                    df_filtrado['nome'].str.upper().str.contains(texto, na=False)
                )
                df_filtrado = df_filtrado[mask]
            
            # Verificar liquidez se solicitado
            if verificar_liquidez:
                tickers_verificar = df_filtrado['ticker'].tolist()
                
                if len(tickers_verificar) > 0:
                    st.info(f"🔍 Verificando liquidez de {len(tickers_verificar)} ativos...")
                    
                    liquidez_info = verificar_liquidez_ativos(tickers_verificar, min_sessoes)
                    
                    # Adicionar informações ao DataFrame
                    df_filtrado['verificado_30d'] = True
                    df_filtrado['negociado_30d'] = df_filtrado['ticker'].map(
                        lambda t: liquidez_info.get(t, {}).get('negociado', False)
                    )
                    df_filtrado['sessoes_ativas'] = df_filtrado['ticker'].map(
                        lambda t: liquidez_info.get(t, {}).get('sessoes', 0)
                    )
                    
                    # Filtrar apenas negociados se solicitado
                    if apenas_negociados:
                        df_filtrado = df_filtrado[df_filtrado['negociado_30d']]
                    
                    st.success(f"✅ Verificação concluída!")
            else:
                df_filtrado['verificado_30d'] = False
                df_filtrado['sessoes_ativas'] = 0
            
            # Salvar resultado
            st.session_state.df_filtrado = df_filtrado
            st.success(f"✅ Filtros aplicados: **{len(df_filtrado)} ativos** encontrados")
    
    # Usar DataFrame filtrado
    df_filtrado = st.session_state.df_filtrado
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Universo Total", len(df_universo))
    
    with col2:
        st.metric("Após Filtros", len(df_filtrado))
    
    with col3:
        st.metric("Selecionados", len(st.session_state.selected_tickers))
    
    with col4:
        if len(df_filtrado) > 50:
            st.warning(f"⚠️ Muitos ativos")
        else:
            st.success(f"✅ OK")
    
    st.markdown("---")
    
    # Seleção rápida
    st.subheader("⚡ Seleção Rápida")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🔝 Top Liquidez", use_container_width=True):
            selecionados = selecionar_por_criterio(df_filtrado, 'liquidez', 10)
            st.session_state.selected_tickers = selecionados
            st.rerun()
    
    with col2:
        if st.button("💰 Top DY", use_container_width=True):
            selecionados = selecionar_por_criterio(df_filtrado, 'dy', 10)
            st.session_state.selected_tickers = selecionados
            st.rerun()
    
    with col3:
        if st.button("🏢 Apenas FIIs", use_container_width=True):
            selecionados = selecionar_por_criterio(df_filtrado, 'fiis', 15)
            st.session_state.selected_tickers = selecionados
            st.rerun()
    
    with col4:
        if st.button("📋 Todos", use_container_width=True):
            if len(df_filtrado) > 50:
                st.warning("⚠️ Muitos ativos! Use filtros para reduzir")
            else:
                st.session_state.selected_tickers = df_filtrado['ticker'].tolist()
                st.rerun()
    
    with col5:
        if st.button("🗑️ Limpar", use_container_width=True):
            st.session_state.selected_tickers = []
            st.rerun()
    
    st.markdown("---")
    
    # Tabela interativa
    st.subheader("📋 Ativos Disponíveis")
    
    if not df_filtrado.empty:
        # Preparar DataFrame para exibição
        df_display = df_filtrado.copy()
        df_display['✓'] = df_display['ticker'].isin(st.session_state.selected_tickers)
        
        # Colunas para exibir
        if 'sessoes_ativas' in df_display.columns:
            cols_display = ['✓', 'ticker', 'nome', 'tipo', 'setor', 'segmento', 'sessoes_ativas']
        else:
            cols_display = ['✓', 'ticker', 'nome', 'tipo', 'setor', 'segmento']
        
        df_display = df_display[cols_display]
        
        # Editor
        edited_df = st.data_editor(
            df_display,
            column_config={
                "✓": st.column_config.CheckboxColumn(
                    "Selecionar",
                    default=False,
                    width="small"
                ),
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "nome": st.column_config.TextColumn("Nome", width="medium"),
                "tipo": st.column_config.TextColumn("Tipo", width="small"),
                "setor": st.column_config.TextColumn("Setor", width="medium"),
                "segmento": st.column_config.TextColumn("Segmento", width="medium"),
                "sessoes_ativas": st.column_config.NumberColumn(
                    "Sessões 30d",
                    help="Dias com negociação",
                    width="small"
                )
            },
            disabled=[c for c in cols_display if c != '✓'],
            hide_index=True,
            use_container_width=True,
            height=500
        )
        
        # Atualizar seleção
        novos_selecionados = edited_df[edited_df['✓']]['ticker'].tolist()
        if set(novos_selecionados) != set(st.session_state.selected_tickers):
            st.session_state.selected_tickers = novos_selecionados
            st.rerun()
    
    else:
        st.warning("⚠️ Nenhum ativo encontrado. Ajuste os filtros.")
    
    st.markdown("---")
    
    # Resumo da seleção
    st.subheader("✅ Resumo da Seleção")
    
    if st.session_state.selected_tickers:
        df_selecionados = df_filtrado[
            df_filtrado['ticker'].isin(st.session_state.selected_tickers)
        ]
        
        # Estatísticas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total", len(df_selecionados))
        
        with col2:
            num_acoes = len(df_selecionados[df_selecionados['tipo'] == 'ACAO'])
            st.metric("Ações", num_acoes)
        
        with col3:
            num_fiis = len(df_selecionados[df_selecionados['tipo'] == 'FII'])
            st.metric("FIIs", num_fiis)
        
        with col4:
            num_etfs = len(df_selecionados[df_selecionados['tipo'] == 'ETF'])
            st.metric("ETFs", num_etfs)
        
        # Tabela resumida
        st.dataframe(
            df_selecionados[['ticker', 'nome', 'tipo', 'setor']],
            use_container_width=True,
            hide_index=True
        )
        
        # Ações
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Salvar no Portfólio", type="primary", use_container_width=True):
                st.session_state.portfolio_tickers = st.session_state.selected_tickers.copy()
                st.success(f"✅ **{len(st.session_state.portfolio_tickers)} ativos** salvos!")
                st.balloons()
        
        with col2:
            csv = df_selecionados.to_csv(index=False)
            st.download_button(
                "📥 Exportar CSV",
                data=csv,
                file_name=f"ativos_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("ℹ️ Nenhum ativo selecionado. Use a tabela ou botões de seleção rápida.")
    
    # Informações
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        ### 📊 Seleção de Ativos
        
        **1. Configurar Filtros (Sidebar)**
        - Escolha tipo de ativo (Ações, FIIs, ETFs)
        - Filtre por setor e segmento
        - Busque por ticker ou nome
        - Opcionalmente, verifique liquidez (30 dias)
        
        **2. Aplicar Filtros**
        - Clique em "Aplicar Filtros" para executar
        - Aguarde o processamento
        
        **3. Selecionar Ativos**
        - Use botões de seleção rápida, ou
        - Marque manualmente na tabela
        
        **4. Salvar**
        - "Salvar no Portfólio" para usar nas outras páginas
        - "Exportar CSV" para backup
        
        ### 💡 Dicas
        - Arquivo CSV: Coloque seu `b3_universe.csv` em `assets/`
        - Formato CSV: ticker, nome, setor, segmento, tipo
        - Liquidez: Verificação pode levar tempo com muitos ativos
        - Performance: Evite selecionar mais de 50 ativos
        """)


if __name__ == "__main__":
    main()
