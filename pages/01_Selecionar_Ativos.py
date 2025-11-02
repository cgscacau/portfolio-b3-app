"""
📊 Seleção de Ativos
Lista de ativos da B3 com filtros avançados por setor, segmento e liquidez
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
    Carrega universo de ativos do CSV ou cria padrão
    
    Returns:
        DataFrame com [ticker, nome, setor, segmento, tipo]
    """
    csv_path = root_dir / 'assets' / 'b3_universe.csv'
    
    # Tentar carregar do CSV
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Validar colunas
            required_cols = ['ticker', 'nome', 'setor', 'segmento', 'tipo']
            if all(col in df.columns for col in required_cols):
                # Limpar dados
                df['ticker'] = df['ticker'].str.upper().str.strip()
                df['nome'] = df['nome'].str.strip()
                df['setor'] = df['setor'].str.strip()
                df['segmento'] = df['segmento'].str.strip()
                df['tipo'] = df['tipo'].str.upper().str.strip()
                
                # Remover duplicatas e nulos
                df = df.dropna(subset=['ticker'])
                df = df.drop_duplicates(subset=['ticker'])
                
                return df
        except Exception as e:
            st.warning(f"⚠️ Erro ao ler CSV: {str(e)}")
    
    # Criar universo padrão
    return criar_universo_padrao()


def criar_universo_padrao():
    """Cria universo padrão caso CSV não exista"""
    
    ativos = []
    
    # FINANCEIRO
    ativos.extend([
        ('ITUB4', 'Itaú Unibanco PN', 'Financeiro', 'Bancos', 'ACAO'),
        ('BBDC4', 'Bradesco PN', 'Financeiro', 'Bancos', 'ACAO'),
        ('BBAS3', 'Banco do Brasil ON', 'Financeiro', 'Bancos', 'ACAO'),
        ('SANB11', 'Santander Units', 'Financeiro', 'Bancos', 'ACAO'),
        ('BBSE3', 'BB Seguridade ON', 'Financeiro', 'Seguros', 'ACAO'),
        ('PSSA3', 'Porto Seguro ON', 'Financeiro', 'Seguros', 'ACAO'),
        ('B3SA3', 'B3 ON', 'Financeiro', 'Serviços Financeiros', 'ACAO'),
    ])
    
    # ENERGIA
    ativos.extend([
        ('PETR4', 'Petrobras PN', 'Energia', 'Petróleo e Gás', 'ACAO'),
        ('PETR3', 'Petrobras ON', 'Energia', 'Petróleo e Gás', 'ACAO'),
        ('PRIO3', 'PetroRio ON', 'Energia', 'Petróleo e Gás', 'ACAO'),
        ('RRRP3', '3R Petroleum ON', 'Energia', 'Petróleo e Gás', 'ACAO'),
        ('ELET3', 'Eletrobras ON', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('ELET6', 'Eletrobras PNB', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('ENBR3', 'Energias BR ON', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('ENEV3', 'Eneva ON', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('CPFE3', 'CPFL Energia ON', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('CMIG4', 'Cemig PN', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('TAEE11', 'Taesa Units', 'Energia', 'Energia Elétrica', 'ACAO'),
        ('TRPL4', 'Transmissão Paulista PN', 'Energia', 'Energia Elétrica', 'ACAO'),
    ])
    
    # MATERIAIS BÁSICOS
    ativos.extend([
        ('VALE3', 'Vale ON', 'Materiais Básicos', 'Mineração', 'ACAO'),
        ('CSNA3', 'CSN ON', 'Materiais Básicos', 'Siderurgia', 'ACAO'),
        ('GGBR4', 'Gerdau PN', 'Materiais Básicos', 'Siderurgia', 'ACAO'),
        ('GOAU4', 'Gerdau Met PN', 'Materiais Básicos', 'Siderurgia', 'ACAO'),
        ('USIM5', 'Usiminas PNA', 'Materiais Básicos', 'Siderurgia', 'ACAO'),
        ('SUZB3', 'Suzano ON', 'Materiais Básicos', 'Papel e Celulose', 'ACAO'),
    ])
    
    # CONSUMO
    ativos.extend([
        ('ABEV3', 'Ambev ON', 'Consumo', 'Bebidas', 'ACAO'),
        ('SMTO3', 'São Martinho ON', 'Consumo', 'Alimentos', 'ACAO'),
        ('BEEF3', 'Minerva ON', 'Consumo', 'Alimentos', 'ACAO'),
        ('JBSS3', 'JBS ON', 'Consumo', 'Alimentos', 'ACAO'),
        ('MRFG3', 'Marfrig ON', 'Consumo', 'Alimentos', 'ACAO'),
        ('PCAR3', 'GPA ON', 'Consumo', 'Varejo', 'ACAO'),
        ('LREN3', 'Lojas Renner ON', 'Consumo', 'Varejo', 'ACAO'),
        ('MGLU3', 'Magazine Luiza ON', 'Consumo', 'Varejo', 'ACAO'),
        ('VIIA3', 'Via ON', 'Consumo', 'Varejo', 'ACAO'),
        ('CRFB3', 'Carrefour Brasil ON', 'Consumo', 'Varejo', 'ACAO'),
        ('ASAI3', 'Assaí ON', 'Consumo', 'Varejo', 'ACAO'),
    ])
    
    # SAÚDE
    ativos.extend([
        ('RADL3', 'Raia Drogasil ON', 'Saúde', 'Farmácias', 'ACAO'),
        ('PNVL3', 'Dasa ON', 'Saúde', 'Serviços Médicos', 'ACAO'),
        ('HAPV3', 'Hapvida ON', 'Saúde', 'Operadoras de Saúde', 'ACAO'),
        ('FLRY3', 'Fleury ON', 'Saúde', 'Serviços Médicos', 'ACAO'),
    ])
    
    # INDUSTRIAL
    ativos.extend([
        ('WEGE3', 'WEG ON', 'Industrial', 'Máquinas e Equipamentos', 'ACAO'),
        ('EMBR3', 'Embraer ON', 'Industrial', 'Aeronáutica', 'ACAO'),
        ('RAIZ4', 'Raízen PN', 'Industrial', 'Combustíveis', 'ACAO'),
        ('RAIL3', 'Rumo ON', 'Industrial', 'Transporte', 'ACAO'),
        ('CCRO3', 'CCR ON', 'Industrial', 'Concessões', 'ACAO'),
        ('CPLE6', 'Copel PNB', 'Industrial', 'Energia', 'ACAO'),
    ])
    
    # TECNOLOGIA E TELECOM
    ativos.extend([
        ('VIVT3', 'Vivo ON', 'Tecnologia', 'Telecomunicações', 'ACAO'),
        ('TIMS3', 'Tim ON', 'Tecnologia', 'Telecomunicações', 'ACAO'),
        ('OIBR3', 'Oi ON', 'Tecnologia', 'Telecomunicações', 'ACAO'),
        ('TOTS3', 'Totvs ON', 'Tecnologia', 'Software', 'ACAO'),
        ('LWSA3', 'Locaweb ON', 'Tecnologia', 'Internet', 'ACAO'),
    ])
    
    # IMOBILIÁRIO
    ativos.extend([
        ('CYRE3', 'Cyrela ON', 'Imobiliário', 'Construção', 'ACAO'),
        ('MRVE3', 'MRV ON', 'Imobiliário', 'Construção', 'ACAO'),
        ('EZTC3', 'EzTec ON', 'Imobiliário', 'Construção', 'ACAO'),
        ('RENT3', 'Localiza ON', 'Imobiliário', 'Aluguel de Veículos', 'ACAO'),
    ])
    
    # UTILIDADES
    ativos.extend([
        ('SBSP3', 'Sabesp ON', 'Utilidades', 'Água e Saneamento', 'ACAO'),
        ('CSMG3', 'Copasa ON', 'Utilidades', 'Água e Saneamento', 'ACAO'),
    ])
    
    # EDUCAÇÃO
    ativos.extend([
        ('YDUQ3', 'Yduqs ON', 'Educação', 'Educação', 'ACAO'),
        ('COGN3', 'Cogna ON', 'Educação', 'Educação', 'ACAO'),
    ])
    
    # FIIs - FUNDOS IMOBILIÁRIOS
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
        ('HGRE11', 'CSHG Real Estate', 'Fundos Imobiliários', 'Multiestratégia', 'FII'),
        ('BCFF11', 'BTG Fundo de Fundos', 'Fundos Imobiliários', 'Fundo de Fundos', 'FII'),
        ('RZTR11', 'Riza Terrax', 'Fundos Imobiliários', 'Desenvolvimento', 'FII'),
        ('VILG11', 'Vinci Logística', 'Fundos Imobiliários', 'Logística', 'FII'),
        ('BRCO11', 'Bresco Logística', 'Fundos Imobiliários', 'Logística', 'FII'),
    ])
    
    # ETFs
    ativos.extend([
        ('BOVA11', 'Ibovespa ETF', 'ETFs', 'Índice', 'ETF'),
        ('SMAL11', 'Small Caps ETF', 'ETFs', 'Índice', 'ETF'),
        ('IVVB11', 'S&P 500 ETF', 'ETFs', 'Índice', 'ETF'),
        ('PIBB11', 'IBrX ETF', 'ETFs', 'Índice', 'ETF'),
        ('HASH11', 'Nasdaq Crypto ETF', 'ETFs', 'Criptomoedas', 'ETF'),
    ])
    
    df = pd.DataFrame(ativos, columns=['ticker', 'nome', 'setor', 'segmento', 'tipo'])
    return df


# ==========================================
# VERIFICAÇÃO DE LIQUIDEZ
# ==========================================

def verificar_liquidez_batch(tickers, min_sessoes=5):
    """
    Verifica liquidez de múltiplos ativos
    
    Args:
        tickers: Lista de tickers
        min_sessoes: Mínimo de sessões ativas
        
    Returns:
        Dict com informações de liquidez
    """
    resultado = {}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=35)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Verificando liquidez: {ticker} ({idx+1}/{total})")
        
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
            
        except Exception as e:
            resultado[ticker] = {
                'negociado': False,
                'sessoes': 0
            }
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    return resultado


# ==========================================
# SELEÇÃO INTELIGENTE
# ==========================================

def selecionar_top_liquidez(df, n=10):
    """Seleciona ativos mais líquidos (blue chips)"""
    blue_chips = [
        'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 
        'ABEV3', 'WEGE3', 'B3SA3', 'RENT3', 'ELET3',
        'SUZB3', 'RAIL3', 'ENBR3', 'RADL3', 'VIVT3'
    ]
    return [t for t in blue_chips if t in df['ticker'].values][:n]


def selecionar_top_dy(df, n=10):
    """Seleciona ativos com histórico de bons dividendos"""
    alto_dy = [
        'ITUB4', 'BBDC4', 'BBAS3', 'PETR4', 'VALE3', 
        'TAEE11', 'TRPL4', 'CPFE3', 'CMIG4', 'CPLE6'
    ]
    return [t for t in alto_dy if t in df['ticker'].values][:n]


def selecionar_fiis(df, n=15):
    """Seleciona apenas FIIs"""
    fiis = df[df['tipo'] == 'FII']['ticker'].head(n).tolist()
    return fiis


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal"""
    
    st.title("📊 Seleção de Ativos")
    st.markdown("Selecione ativos da B3 para análise de portfólio e dividendos")
    st.markdown("---")
    
    # Carregar universo
    with st.spinner("📥 Carregando universo B3..."):
        df_universo = carregar_universo_b3()
    
    # Verificar origem
    csv_path = root_dir / 'assets' / 'b3_universe.csv'
    if csv_path.exists():
        st.success(f"✅ **{len(df_universo)} ativos** carregados do arquivo `b3_universe.csv`")
    else:
        st.info(f"ℹ️ **{len(df_universo)} ativos** no universo padrão")
        st.caption("💡 Crie `assets/b3_universe.csv` para personalizar (colunas: ticker, nome, setor, segmento, tipo)")
    
    # Inicializar estado de filtros
    if 'df_filtrado' not in st.session_state:
        st.session_state.df_filtrado = df_universo.copy()
    
    # Sidebar - Filtros
    with st.sidebar:
        st.header("🔍 Filtros")
        
        # Tipo
        st.subheader("📋 Tipo de Ativo")
        tipos_disponiveis = sorted(df_universo['tipo'].unique())
        tipos_selecionados = st.multiselect(
            "Tipos",
            options=tipos_disponiveis,
            default=tipos_disponiveis,
            help="Ações, FIIs ou ETFs"
        )
        
        st.markdown("---")
        
        # Setor
        st.subheader("🏢 Setor")
        setores_disponiveis = ['Todos'] + sorted(df_universo['setor'].unique())
        setor_selecionado = st.selectbox(
            "Setor",
            options=setores_disponiveis
        )
        
        # Segmento (dependente do setor)
        st.subheader("🎯 Segmento")
        if setor_selecionado != 'Todos':
            segmentos_filtrados = sorted(
                df_universo[df_universo['setor'] == setor_selecionado]['segmento'].unique()
            )
        else:
            segmentos_filtrados = sorted(df_universo['segmento'].unique())
        
        segmentos_disponiveis = ['Todos'] + segmentos_filtrados
        segmento_selecionado = st.selectbox(
            "Segmento",
            options=segmentos_disponiveis
        )
        
        st.markdown("---")
        
        # Busca
        st.subheader("🔎 Busca")
        texto_busca = st.text_input(
            "Ticker ou Nome",
            placeholder="Ex: PETR4, Petrobras"
        )
        
        st.markdown("---")
        
        # Liquidez
        st.subheader("📈 Liquidez (30 dias)")
        
        verificar_liquidez = st.checkbox(
            "Verificar negociação",
            value=False,
            help="Verifica ativos negociados nos últimos 30 dias (pode demorar)"
        )
        
        if verificar_liquidez:
            min_sessoes = st.slider(
                "Mínimo de sessões",
                min_value=1,
                max_value=20,
                value=5
            )
            
            apenas_negociados = st.checkbox(
                "Apenas negociados",
                value=True
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
    
    # Aplicar filtros
    if btn_aplicar:
        
        with st.spinner("🔍 Aplicando filtros..."):
            df_filtrado = df_universo.copy()
            
            # Tipo
            if tipos_selecionados:
                df_filtrado = df_filtrado[df_filtrado['tipo'].isin(tipos_selecionados)]
            
            # Setor
            if setor_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['setor'] == setor_selecionado]
            
            # Segmento
            if segmento_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['segmento'] == segmento_selecionado]
            
            # Busca
            if texto_busca:
                texto = texto_busca.upper()
                mask = (
                    df_filtrado['ticker'].str.contains(texto, na=False) |
                    df_filtrado['nome'].str.upper().str.contains(texto, na=False)
                )
                df_filtrado = df_filtrado[mask]
            
            # Liquidez
            if verificar_liquidez and len(df_filtrado) > 0:
                tickers_verificar = df_filtrado['ticker'].tolist()
                
                st.info(f"🔍 Verificando liquidez de {len(tickers_verificar)} ativos...")
                liquidez_info = verificar_liquidez_batch(tickers_verificar, min_sessoes)
                
                df_filtrado['negociado_30d'] = df_filtrado['ticker'].map(
                    lambda t: liquidez_info.get(t, {}).get('negociado', False)
                )
                df_filtrado['sessoes_ativas'] = df_filtrado['ticker'].map(
                    lambda t: liquidez_info.get(t, {}).get('sessoes', 0)
                )
                
                if apenas_negociados:
                    df_filtrado = df_filtrado[df_filtrado['negociado_30d']]
                
                st.success("✅ Verificação concluída!")
            else:
                df_filtrado['sessoes_ativas'] = 0
            
            st.session_state.df_filtrado = df_filtrado
            st.success(f"✅ **{len(df_filtrado)} ativos** encontrados")
    
    df_filtrado = st.session_state.df_filtrado
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Universo", len(df_universo))
    
    with col2:
        st.metric("Filtrados", len(df_filtrado))
    
    with col3:
        st.metric("Selecionados", len(st.session_state.selected_tickers))
    
    with col4:
        if len(df_filtrado) > 100:
            st.warning("⚠️ Muitos ativos")
        elif len(df_filtrado) > 50:
            st.info("ℹ️ Moderado")
        else:
            st.success("✅ Ótimo")
    
    st.markdown("---")
    
    # Seleção rápida
    st.subheader("⚡ Seleção Rápida")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🔝 Top Liquidez", use_container_width=True):
            st.session_state.selected_tickers = selecionar_top_liquidez(df_filtrado, 10)
            st.rerun()
    
    with col2:
        if st.button("💰 Top DY", use_container_width=True):
            st.session_state.selected_tickers = selecionar_top_dy(df_filtrado, 10)
            st.rerun()
    
    with col3:
        if st.button("🏢 FIIs", use_container_width=True):
            st.session_state.selected_tickers = selecionar_fiis(df_filtrado, 15)
            st.rerun()
    
    with col4:
        # CORREÇÃO: Sempre permitir selecionar todos
        if st.button("📋 Selecionar Todos", use_container_width=True):
            st.session_state.selected_tickers = df_filtrado['ticker'].tolist()
            st.rerun()
    
    with col5:
        if st.button("🗑️ Limpar", use_container_width=True):
            st.session_state.selected_tickers = []
            st.rerun()
    
    # Aviso se muitos ativos
    if len(df_filtrado) > 50:
        st.info(f"💡 **{len(df_filtrado)} ativos** disponíveis. Use filtros para reduzir e melhorar performance das análises.")
    
    st.markdown("---")
    
    # Tabela
    st.subheader("📋 Ativos Disponíveis")
    
    if not df_filtrado.empty:
        df_display = df_filtrado.copy()
        df_display['✓'] = df_display['ticker'].isin(st.session_state.selected_tickers)
        
        # Colunas
        if 'sessoes_ativas' in df_display.columns:
            cols = ['✓', 'ticker', 'nome', 'tipo', 'setor', 'segmento', 'sessoes_ativas']
        else:
            cols = ['✓', 'ticker', 'nome', 'tipo', 'setor', 'segmento']
        
        df_display = df_display[cols]
        
        # Editor
        edited_df = st.data_editor(
            df_display,
            column_config={
                "✓": st.column_config.CheckboxColumn("Sel", default=False, width="small"),
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "nome": st.column_config.TextColumn("Nome", width="medium"),
                "tipo": st.column_config.TextColumn("Tipo", width="small"),
                "setor": st.column_config.TextColumn("Setor", width="medium"),
                "segmento": st.column_config.TextColumn("Segmento", width="medium"),
                "sessoes_ativas": st.column_config.NumberColumn("Sessões 30d", width="small")
            },
            disabled=[c for c in cols if c != '✓'],
            hide_index=True,
            use_container_width=True,
            height=500
        )
        
        # Atualizar seleção
        novos = edited_df[edited_df['✓']]['ticker'].tolist()
        if set(novos) != set(st.session_state.selected_tickers):
            st.session_state.selected_tickers = novos
            st.rerun()
    
    else:
        st.warning("⚠️ Nenhum ativo encontrado. Ajuste os filtros.")
    
    st.markdown("---")
    
    # Resumo
    st.subheader("✅ Resumo da Seleção")
    
    if st.session_state.selected_tickers:
        df_sel = df_filtrado[df_filtrado['ticker'].isin(st.session_state.selected_tickers)]
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total", len(df_sel))
        with col2:
            st.metric("Ações", len(df_sel[df_sel['tipo'] == 'ACAO']))
        with col3:
            st.metric("FIIs", len(df_sel[df_sel['tipo'] == 'FII']))
        with col4:
            st.metric("ETFs", len(df_sel[df_sel['tipo'] == 'ETF']))
        
        # Tabela
        st.dataframe(
            df_sel[['ticker', 'nome', 'tipo', 'setor']],
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
            csv = df_sel.to_csv(index=False)
            st.download_button(
                "📥 Exportar CSV",
                data=csv,
                file_name=f"ativos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("ℹ️ Nenhum ativo selecionado. Use os botões acima ou marque na tabela.")
    
    # Info
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        ### 📊 Guia Rápido
        
        **1. Configurar Filtros (Sidebar)**
        - Tipo: Ações, FIIs, ETFs
        - Setor e Segmento
        - Busca por nome/ticker
        - Liquidez (opcional, pode demorar)
        
        **2. Aplicar Filtros**
        - Clique em "Aplicar Filtros"
        
        **3. Selecionar**
        - Botões rápidos (Top Liquidez, Top DY, FIIs, Todos)
        - Ou marque manualmente na tabela
        
        **4. Salvar**
        - "Salvar no Portfólio" para usar nas análises
        - "Exportar CSV" para backup
        
        ### 📁 Arquivo CSV Personalizado
        
        Crie `assets/b3_universe.csv` com:
        ```
        ticker,nome,setor,segmento,tipo
        PETR4,Petrobras PN,Energia,Petróleo e Gás,ACAO
        VALE3,Vale ON,Materiais Básicos,Mineração,ACAO
        HGLG11,CSHG Logística,Fundos Imobiliários,Logística,FII
        ```
        """)


if __name__ == "__main__":
    main()
