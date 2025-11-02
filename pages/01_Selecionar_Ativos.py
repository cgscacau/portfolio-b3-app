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
# UNIVERSO BASE DE ATIVOS B3
# ==========================================

@st.cache_data(ttl=86400, show_spinner=False)
def carregar_universo_b3():
    """
    Carrega universo completo de ativos da B3
    
    Returns:
        DataFrame com [ticker, nome, setor, segmento, tipo]
    """
    # Lista expandida de ativos B3 por setor
    ativos = {
        'ticker': [],
        'nome': [],
        'setor': [],
        'segmento': [],
        'tipo': []
    }
    
    # SETOR FINANCEIRO
    financeiro = [
        ('ITUB4', 'Itaú Unibanco', 'Bancos'),
        ('BBDC4', 'Bradesco', 'Bancos'),
        ('BBAS3', 'Banco do Brasil', 'Bancos'),
        ('SANB11', 'Santander', 'Bancos'),
        ('BBSE3', 'BB Seguridade', 'Seguros'),
        ('PSSA3', 'Porto Seguro', 'Seguros'),
        ('CSAN3', 'Cosan', 'Holding'),
        ('B3SA3', 'B3', 'Serviços Financeiros'),
    ]
    
    # SETOR ENERGIA
    energia = [
        ('PETR4', 'Petrobras', 'Petróleo e Gás'),
        ('PETR3', 'Petrobras', 'Petróleo e Gás'),
        ('PRIO3', 'PetroRio', 'Petróleo e Gás'),
        ('RRRP3', '3R Petroleum', 'Petróleo e Gás'),
        ('ELET3', 'Eletrobras', 'Energia Elétrica'),
        ('ELET6', 'Eletrobras', 'Energia Elétrica'),
        ('ENBR3', 'Energias BR', 'Energia Elétrica'),
        ('ENEV3', 'Eneva', 'Energia Elétrica'),
        ('CPFE3', 'CPFL Energia', 'Energia Elétrica'),
        ('CMIG4', 'Cemig', 'Energia Elétrica'),
        ('TAEE11', 'Taesa', 'Energia Elétrica'),
        ('TRPL4', 'Transmissão Paulista', 'Energia Elétrica'),
    ]
    
    # SETOR MATERIAIS BÁSICOS
    materiais = [
        ('VALE3', 'Vale', 'Mineração'),
        ('CSNA3', 'CSN', 'Siderurgia'),
        ('GGBR4', 'Gerdau', 'Siderurgia'),
        ('GOAU4', 'Gerdau Metalúrgica', 'Siderurgia'),
        ('USIM5', 'Usiminas', 'Siderurgia'),
        ('SUZB3', 'Suzano', 'Papel e Celulose'),
    ]
    
    # SETOR CONSUMO
    consumo = [
        ('ABEV3', 'Ambev', 'Bebidas'),
        ('SMTO3', 'São Martinho', 'Alimentos'),
        ('BEEF3', 'Minerva', 'Alimentos'),
        ('JBSS3', 'JBS', 'Alimentos'),
        ('MRFG3', 'Marfrig', 'Alimentos'),
        ('PCAR3', 'GPA', 'Varejo'),
        ('LREN3', 'Lojas Renner', 'Varejo'),
        ('AMER3', 'Lojas Americanas', 'Varejo'),
        ('MGLU3', 'Magazine Luiza', 'Varejo'),
        ('VIIA3', 'Via', 'Varejo'),
        ('CRFB3', 'Carrefour Brasil', 'Varejo'),
        ('ASAI3', 'Assaí', 'Varejo'),
    ]
    
    # SETOR SAÚDE
    saude = [
        ('RADL3', 'Raia Drogasil', 'Farmácias'),
        ('PNVL3', 'Dasa', 'Serviços Médicos'),
        ('HAPV3', 'Hapvida', 'Saúde'),
        ('FLRY3', 'Fleury', 'Serviços Médicos'),
    ]
    
    # SETOR INDUSTRIAL
    industrial = [
        ('WEGE3', 'WEG', 'Máquinas e Equipamentos'),
        ('EMBR3', 'Embraer', 'Aeronáutica'),
        ('RAIZ4', 'Raízen', 'Combustíveis'),
        ('RAIL3', 'Rumo', 'Transporte'),
        ('CCRO3', 'CCR', 'Concessões'),
        ('CPLE6', 'Copel', 'Energia'),
    ]
    
    # SETOR TECNOLOGIA E TELECOM
    tech = [
        ('VIVT3', 'Vivo', 'Telecomunicações'),
        ('TIMS3', 'Tim', 'Telecomunicações'),
        ('OIBR3', 'Oi', 'Telecomunicações'),
        ('TOTS3', 'Totvs', 'Software'),
        ('LWSA3', 'Locaweb', 'Internet'),
    ]
    
    # SETOR IMOBILIÁRIO
    imobiliario = [
        ('CYRE3', 'Cyrela', 'Construção'),
        ('MRVE3', 'MRV', 'Construção'),
        ('EZTC3', 'EzTec', 'Construção'),
        ('RENT3', 'Localiza', 'Aluguel de Veículos'),
    ]
    
    # SETOR UTILIDADES
    utilidades = [
        ('SBSP3', 'Sabesp', 'Água e Saneamento'),
        ('CSMG3', 'Copasa', 'Água e Saneamento'),
    ]
    
    # EDUCAÇÃO
    educacao = [
        ('YDUQ3', 'Yduqs', 'Educação'),
        ('COGN3', 'Cogna', 'Educação'),
    ]
    
    # FIIs - FUNDOS IMOBILIÁRIOS
    fiis = [
        ('HGLG11', 'CSHG Logística', 'Logística'),
        ('MXRF11', 'Maxi Renda', 'Lajes Corporativas'),
        ('KNRI11', 'Kinea Renda', 'Lajes Corporativas'),
        ('XPML11', 'XP Malls', 'Shopping'),
        ('VISC11', 'Vinci Shopping', 'Shopping'),
        ('BTLG11', 'BTG Logística', 'Logística'),
        ('HGRU11', 'CSHG Renda Urbana', 'Multiestratégia'),
        ('KNCR11', 'Kinea Crédito', 'Crédito'),
        ('PVBI11', 'PV Birigui', 'Lajes Corporativas'),
        ('IRDM11', 'Iridium', 'Lajes Corporativas'),
        ('HGRE11', 'CSHG Real Estate', 'Multiestratégia'),
        ('BCFF11', 'BTG Fundo de Fundos', 'Fundo de Fundos'),
        ('RZTR11', 'Riza Terrax', 'Desenvolvimento'),
        ('VILG11', 'Vinci Logística', 'Logística'),
        ('BRCO11', 'Bresco Logística', 'Logística'),
    ]
    
    # ETFs
    etfs = [
        ('BOVA11', 'Ibovespa', 'Índice'),
        ('SMAL11', 'Small Caps', 'Índice'),
        ('IVVB11', 'S&P 500', 'Índice'),
        ('PIBB11', 'IBrX', 'Índice'),
        ('HASH11', 'Nasdaq Crypto', 'Criptomoedas'),
    ]
    
    # Processar todos os setores
    setores_data = [
        (financeiro, 'Financeiro', 'ACAO'),
        (energia, 'Energia', 'ACAO'),
        (materiais, 'Materiais Básicos', 'ACAO'),
        (consumo, 'Consumo', 'ACAO'),
        (saude, 'Saúde', 'ACAO'),
        (industrial, 'Industrial', 'ACAO'),
        (tech, 'Tecnologia', 'ACAO'),
        (imobiliario, 'Imobiliário', 'ACAO'),
        (utilidades, 'Utilidades', 'ACAO'),
        (educacao, 'Educação', 'ACAO'),
        (fiis, 'Fundos Imobiliários', 'FII'),
        (etfs, 'ETFs', 'ETF'),
    ]
    
    for lista_ativos, setor, tipo in setores_data:
        for ticker, nome, segmento in lista_ativos:
            ativos['ticker'].append(ticker)
            ativos['nome'].append(nome)
            ativos['setor'].append(setor)
            ativos['segmento'].append(segmento)
            ativos['tipo'].append(tipo)
    
    df = pd.DataFrame(ativos)
    return df


# ==========================================
# FILTRO DE NEGOCIAÇÃO (30 DIAS)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def filtrar_negociados_30d(df_universo, min_sessoes=5, min_volume=1000):
    """
    Filtra ativos negociados nos últimos 30 dias
    
    Args:
        df_universo: DataFrame com universo de ativos
        min_sessoes: Mínimo de sessões com volume
        min_volume: Volume mínimo por sessão
        
    Returns:
        DataFrame com coluna adicional 'negociado_30d'
    """
    df = df_universo.copy()
    df['negociado_30d'] = False
    df['volume_medio'] = 0.0
    df['sessoes_ativas'] = 0
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=35)
    
    for idx, row in df.iterrows():
        try:
            ticker = row['ticker']
            
            # Buscar histórico
            hist = data.get_price_history(
                [ticker],
                start_date,
                end_date,
                use_cache=True
            )
            
            if not hist.empty and ticker in hist.columns:
                # Contar sessões com volume (se disponível)
                # Como estamos usando preços, vamos verificar se há dados
                dados_validos = hist[ticker].dropna()
                sessoes_ativas = len(dados_validos)
                
                # Calcular volume médio (se disponível no histórico)
                volume_medio = 0
                
                # Verificar critério
                if sessoes_ativas >= min_sessoes:
                    df.at[idx, 'negociado_30d'] = True
                    df.at[idx, 'sessoes_ativas'] = sessoes_ativas
                    df.at[idx, 'volume_medio'] = volume_medio
            
        except Exception as e:
            continue
    
    return df


# ==========================================
# FUNÇÕES DE SELEÇÃO INTELIGENTE
# ==========================================

def selecionar_top_liquidez(df, n=10):
    """Seleciona top N ativos por liquidez"""
    df_sorted = df[df['negociado_30d']].sort_values('sessoes_ativas', ascending=False)
    return df_sorted.head(n)['ticker'].tolist()


def selecionar_top_dy(df, n=10):
    """Seleciona top N ativos por Dividend Yield estimado"""
    # Para simplificar, vamos priorizar FIIs e ações de dividendos conhecidas
    tickers_alto_dy = [
        'ITUB4', 'BBDC4', 'BBAS3', 'PETR4', 'VALE3', 'TAEE11',
        'HGLG11', 'MXRF11', 'KNRI11', 'XPML11', 'VISC11',
        'BTLG11', 'HGRU11', 'KNCR11', 'PVBI11'
    ]
    
    df_dy = df[df['ticker'].isin(tickers_alto_dy) & df['negociado_30d']]
    return df_dy.head(n)['ticker'].tolist()


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal da página"""
    
    st.title("📊 Seleção de Ativos")
    st.markdown("Lista de ativos negociados nos últimos 30 dias com filtros avançados")
    st.markdown("---")
    
    # Carregar universo
    with st.spinner("📥 Carregando universo de ativos B3..."):
        df_universo = carregar_universo_b3()
        st.success(f"✅ **{len(df_universo)} ativos** no universo B3")
    
    # Sidebar - Filtros
    with st.sidebar:
        st.header("🔍 Filtros")
        
        # Filtro de negociação
        st.subheader("📈 Negociação")
        
        aplicar_filtro_30d = st.checkbox(
            "Apenas negociados (30 dias)",
            value=True,
            help="Filtra apenas ativos com negociação nos últimos 30 dias"
        )
        
        if aplicar_filtro_30d:
            min_sessoes = st.slider(
                "Mínimo de sessões ativas",
                min_value=1,
                max_value=20,
                value=5,
                help="Número mínimo de dias com negociação"
            )
        else:
            min_sessoes = 0
        
        st.markdown("---")
        
        # Filtro por tipo
        st.subheader("📋 Tipo de Ativo")
        
        tipos_disponiveis = sorted(df_universo['tipo'].unique())
        tipos_selecionados = st.multiselect(
            "Selecione os tipos",
            options=tipos_disponiveis,
            default=tipos_disponiveis,
            help="Ações, FIIs ou ETFs"
        )
        
        st.markdown("---")
        
        # Filtro por setor
        st.subheader("🏢 Setor")
        
        setores_disponiveis = sorted(df_universo['setor'].unique())
        setores_selecionados = st.multiselect(
            "Selecione os setores",
            options=setores_disponiveis,
            default=setores_disponiveis,
            help="Filtre por setor econômico"
        )
        
        st.markdown("---")
        
        # Filtro por segmento
        st.subheader("🎯 Segmento")
        
        segmentos_disponiveis = sorted(df_universo['segmento'].unique())
        segmentos_selecionados = st.multiselect(
            "Selecione os segmentos",
            options=segmentos_disponiveis,
            help="Filtre por segmento específico"
        )
        
        st.markdown("---")
        
        # Busca por texto
        st.subheader("🔎 Busca")
        
        texto_busca = st.text_input(
            "Buscar ticker ou nome",
            placeholder="Ex: PETR4, Petrobras...",
            help="Digite parte do código ou nome"
        )
        
        st.markdown("---")
        
        # Botão aplicar filtros
        btn_filtrar = st.button(
            "🔄 Aplicar Filtros",
            type="primary",
            use_container_width=True
        )
    
    # Aplicar filtros
    if btn_filtrar or aplicar_filtro_30d:
        
        # Filtro de negociação 30d
        if aplicar_filtro_30d:
            with st.spinner("🔍 Verificando ativos negociados (isso pode levar alguns minutos)..."):
                df_filtrado = filtrar_negociados_30d(df_universo, min_sessoes)
                df_filtrado = df_filtrado[df_filtrado['negociado_30d']]
        else:
            df_filtrado = df_universo.copy()
            df_filtrado['negociado_30d'] = True
            df_filtrado['sessoes_ativas'] = 0
        
        # Filtro por tipo
        if tipos_selecionados:
            df_filtrado = df_filtrado[df_filtrado['tipo'].isin(tipos_selecionados)]
        
        # Filtro por setor
        if setores_selecionados:
            df_filtrado = df_filtrado[df_filtrado['setor'].isin(setores_selecionados)]
        
        # Filtro por segmento
        if segmentos_selecionados:
            df_filtrado = df_filtrado[df_filtrado['segmento'].isin(segmentos_selecionados)]
        
        # Busca por texto
        if texto_busca:
            texto = texto_busca.upper()
            mask = (
                df_filtrado['ticker'].str.contains(texto, na=False) |
                df_filtrado['nome'].str.upper().str.contains(texto, na=False)
            )
            df_filtrado = df_filtrado[mask]
        
        # Guardar no session state
        st.session_state.universe_df = df_filtrado
    
    else:
        # Usar universo completo se não filtrou
        if st.session_state.universe_df.empty:
            st.session_state.universe_df = df_universo
            st.session_state.universe_df['negociado_30d'] = False
            st.session_state.universe_df['sessoes_ativas'] = 0
        
        df_filtrado = st.session_state.universe_df
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total no Universo", len(df_universo))
    
    with col2:
        st.metric("Após Filtros", len(df_filtrado))
    
    with col3:
        st.metric("Selecionados", len(st.session_state.selected_tickers))
    
    with col4:
        if len(df_filtrado) > 50:
            st.warning(f"⚠️ {len(df_filtrado)} ativos")
        else:
            st.success(f"✅ {len(df_filtrado)} ativos")
    
    st.markdown("---")
    
    # Botões de seleção inteligente
    st.subheader("⚡ Seleção Rápida")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔝 Top 10 Liquidez", use_container_width=True):
            top_liq = selecionar_top_liquidez(df_filtrado, 10)
            st.session_state.selected_tickers = top_liq
            st.success(f"✅ {len(top_liq)} ativos selecionados")
            st.rerun()
    
    with col2:
        if st.button("💰 Top 10 DY", use_container_width=True):
            top_dy = selecionar_top_dy(df_filtrado, 10)
            st.session_state.selected_tickers = top_dy
            st.success(f"✅ {len(top_dy)} ativos selecionados")
            st.rerun()
    
    with col3:
        if st.button("📋 Selecionar Todos", use_container_width=True):
            if len(df_filtrado) > 50:
                st.warning("⚠️ Muitos ativos! Recomendado: use filtros para reduzir")
            else:
                st.session_state.selected_tickers = df_filtrado['ticker'].tolist()
                st.success(f"✅ {len(df_filtrado)} ativos selecionados")
                st.rerun()
    
    with col4:
        if st.button("🗑️ Limpar Seleção", use_container_width=True):
            st.session_state.selected_tickers = []
            st.rerun()
    
    st.markdown("---")
    
    # Tabela de seleção
    st.subheader("📋 Ativos Disponíveis")
    
    if not df_filtrado.empty:
        # Adicionar coluna de seleção
        df_display = df_filtrado.copy()
        df_display['✓'] = df_display['ticker'].isin(st.session_state.selected_tickers)
        
        # Reordenar colunas
        cols_order = ['✓', 'ticker', 'nome', 'tipo', 'setor', 'segmento', 'sessoes_ativas']
        df_display = df_display[cols_order]
        
        # Editor de dados
        edited_df = st.data_editor(
            df_display,
            column_config={
                "✓": st.column_config.CheckboxColumn(
                    "Selecionar",
                    help="Marque para adicionar ao portfólio",
                    default=False,
                    width="small"
                ),
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "nome": st.column_config.TextColumn("Nome", width="medium"),
                "tipo": st.column_config.TextColumn("Tipo", width="small"),
                "setor": st.column_config.TextColumn("Setor", width="medium"),
                "segmento": st.column_config.TextColumn("Segmento", width="medium"),
                "sessoes_ativas": st.column_config.NumberColumn(
                    "Sessões (30d)",
                    help="Dias com negociação nos últimos 30 dias",
                    width="small"
                )
            },
            disabled=["ticker", "nome", "tipo", "setor", "segmento", "sessoes_ativas"],
            hide_index=True,
            use_container_width=True,
            height=500
        )
        
        # Atualizar seleção
        novos_selecionados = edited_df[edited_df['✓']]['ticker'].tolist()
        if novos_selecionados != st.session_state.selected_tickers:
            st.session_state.selected_tickers = novos_selecionados
            st.rerun()
    
    else:
        st.warning("⚠️ Nenhum ativo encontrado com os filtros aplicados")
    
    st.markdown("---")
    
    # Ativos selecionados
    st.subheader("✅ Ativos Selecionados para o Portfólio")
    
    if st.session_state.selected_tickers:
        df_selecionados = df_filtrado[
            df_filtrado['ticker'].isin(st.session_state.selected_tickers)
        ].copy()
        
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
            df_selecionados[['ticker', 'nome', 'tipo', 'setor', 'segmento']],
            use_container_width=True,
            hide_index=True
        )
        
        # Botões de ação
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            if st.button("💾 Salvar no Portfólio", type="primary", use_container_width=True):
                st.session_state.portfolio_tickers = st.session_state.selected_tickers.copy()
                st.success(f"✅ **{len(st.session_state.portfolio_tickers)} ativos** salvos no portfólio!")
                st.balloons()
        
        with col2:
            # Export CSV
            csv = df_selecionados.to_csv(index=False)
            st.download_button(
                label="📥 Exportar CSV",
                data=csv,
                file_name=f"ativos_selecionados_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("ℹ️ Nenhum ativo selecionado ainda. Use a tabela acima ou os botões de seleção rápida.")
    
    # Informações
    with st.expander("ℹ️ Como usar esta página"):
        st.markdown("""
        ### 📊 Seleção de Ativos
        
        **1. Aplicar Filtros**
        - Use a barra lateral para filtrar por tipo, setor, segmento
        - Ative "Apenas negociados (30 dias)" para liquidez
        - Busque por ticker ou nome específico
        
        **2. Seleção Rápida**
        - **Top 10 Liquidez**: Ativos mais negociados
        - **Top 10 DY**: Ativos com melhor histórico de dividendos
        - **Selecionar Todos**: Todos os ativos filtrados (máx. 50 recomendado)
        
        **3. Seleção Manual**
        - Marque/desmarque ativos na tabela
        - Ordene clicando nos cabeçalhos das colunas
        
        **4. Salvar**
        - Clique em "Salvar no Portfólio" para usar nas outras páginas
        - Exporte para CSV se desejar backup
        
        ### ⚠️ Dicas
        - Evite selecionar mais de 50 ativos (impacta performance)
        - Diversifique entre setores diferentes
        - Priorize ativos com boa liquidez (sessões ativas > 5)
        """)


if __name__ == "__main__":
    main()
