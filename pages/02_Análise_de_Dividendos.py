"""
💰 Análise de Dividendos
Histórico, regularidade e calendário mensal simulado de dividendos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from calendar import month_name
import sys
from pathlib import Path

# Configurar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data
from core.init import init_all
from core.cache import salvar_dados_cache, carregar_dados_cache

# Configuração da página
st.set_page_config(
    page_title="Análise de Dividendos",
    page_icon="💰",
    layout="wide"
)

# Inicializar
init_all()

# Inicializar estados específicos da página
if 'analise_dividendos_completa' not in st.session_state:
    st.session_state.analise_dividendos_completa = False

if 'metricas_dividendos' not in st.session_state:
    st.session_state.metricas_dividendos = None

if 'calendario_dividendos' not in st.session_state:
    st.session_state.calendario_dividendos = None


# ==========================================
# FUNÇÕES DE CÁLCULO
# ==========================================

def calcular_dividend_yield(dividendos_df, preco_medio):
    """
    Calcula Dividend Yield anual
    
    Args:
        dividendos_df: DataFrame com dividendos
        preco_medio: Preço médio do período
        
    Returns:
        Float com DY percentual
    """
    if dividendos_df.empty or preco_medio == 0 or pd.isna(preco_medio):
        return 0.0
    
    total_dividendos = dividendos_df['valor'].sum()
    dy = (total_dividendos / preco_medio) * 100
    
    return dy


def calcular_regularidade(dividendos_mensais):
    """
    Calcula índice de regularidade (0-100)
    Baseado no coeficiente de variação invertido
    
    Args:
        dividendos_mensais: Series com dividendos por mês
        
    Returns:
        Float entre 0 e 100
    """
    if len(dividendos_mensais) < 2:
        return 0.0
    
    # Remover zeros
    divs_nao_zero = dividendos_mensais[dividendos_mensais > 0]
    
    if len(divs_nao_zero) < 2:
        return 0.0
    
    media = divs_nao_zero.mean()
    std = divs_nao_zero.std()
    
    if media == 0:
        return 0.0
    
    # Coeficiente de variação invertido
    cv = std / media
    regularidade = max(0, 100 * (1 - min(cv, 1)))
    
    return regularidade


def agrupar_dividendos_por_mes(dividendos_df):
    """
    Agrupa dividendos por mês
    
    Args:
        dividendos_df: DataFrame com ['data', 'valor']
        
    Returns:
        Series indexada por mês (YYYY-MM)
    """
    if dividendos_df.empty:
        return pd.Series(dtype=float)
    
    df = dividendos_df.copy()
    df['mes'] = pd.to_datetime(df['data']).dt.to_period('M')
    
    dividendos_mensais = df.groupby('mes')['valor'].sum()
    dividendos_mensais.index = dividendos_mensais.index.astype(str)
    
    return dividendos_mensais


def criar_calendario_completo(dividendos_mensais, data_inicio, data_fim):
    """
    Cria calendário com todos os meses preenchidos
    
    Args:
        dividendos_mensais: Series com dividendos
        data_inicio: Data inicial
        data_fim: Data final
        
    Returns:
        Series com todos os meses
    """
    # Criar range de meses
    meses_range = pd.period_range(
        start=data_inicio,
        end=data_fim,
        freq='M'
    )
    
    # Series com zeros
    calendario = pd.Series(0.0, index=meses_range.astype(str))
    
    # Preencher com valores
    for mes, valor in dividendos_mensais.items():
        if mes in calendario.index:
            calendario[mes] = valor
    
    return calendario


# ==========================================
# CARREGAMENTO DE DADOS
# ==========================================

def carregar_dados_com_cache(tickers, data_inicio, data_fim):
    """
    Carrega dados usando cache global
    
    Args:
        tickers: Lista de tickers
        data_inicio: Data inicial
        data_fim: Data final
        
    Returns:
        Tuple (precos_df, dividendos_dict)
    """
    # Tentar cache
    price_data, dividend_data = carregar_dados_cache(tickers, data_inicio, data_fim)
    
    if price_data is not None:
        st.info("📦 Dados carregados do cache")
        return price_data, dividend_data if dividend_data else {}
    
    # Baixar dados
    st.info("📥 Baixando dados do mercado...")
    
    # Preços
    price_data = data.get_price_history(tickers, data_inicio, data_fim, use_cache=True)
    
    # Dividendos
    dividendos_dict = {}
    
    progress_bar = st.progress(0)
    for idx, ticker in enumerate(tickers):
        try:
            divs = data.get_dividends(ticker, data_inicio, data_fim)
            if not divs.empty:
                dividendos_dict[ticker] = divs
        except:
            continue
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    
    # Salvar cache
    salvar_dados_cache(tickers, data_inicio, data_fim, price_data, dividendos_dict)
    
    return price_data, dividendos_dict


# ==========================================
# VISUALIZAÇÕES
# ==========================================

def criar_heatmap_calendario(calendario_carteira):
    """Cria heatmap de dividendos mensais"""
    
    if not calendario_carteira:
        return None
    
    df_heatmap = pd.DataFrame(calendario_carteira)
    
    # Converter para datetime
    df_heatmap.index = pd.to_datetime(df_heatmap.index + '-01')
    df_heatmap = df_heatmap.sort_index()
    
    # Formatar
    df_heatmap.index = df_heatmap.index.strftime('%Y-%m')
    
    fig = go.Figure(data=go.Heatmap(
        z=df_heatmap.values.T,
        x=df_heatmap.index,
        y=df_heatmap.columns,
        colorscale='Blues',
        hovertemplate='%{y}<br>%{x}<br>R$ %{z:.2f}<extra></extra>',
        colorbar=dict(title="R$")
    ))
    
    fig.update_layout(
        title='Calendário de Dividendos por Ativo',
        xaxis_title='Mês',
        yaxis_title='Ativo',
        height=max(400, len(df_heatmap.columns) * 30),
        hovermode='closest'
    )
    
    return fig


def criar_grafico_mensal(dividendos_mensais_total):
    """Cria gráfico de barras mensal"""
    
    if dividendos_mensais_total.empty:
        return None
    
    datas = pd.to_datetime(dividendos_mensais_total.index + '-01')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=datas,
        y=dividendos_mensais_total.values,
        marker_color='#3498db',
        hovertemplate='%{x|%B %Y}<br>R$ %{y:.2f}<extra></extra>'
    ))
    
    # Linha de média
    media = dividendos_mensais_total.mean()
    fig.add_hline(
        y=media,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Média: R$ {media:.2f}',
        annotation_position='right'
    )
    
    fig.update_layout(
        title='Fluxo Mensal de Dividendos da Carteira',
        xaxis_title='Mês',
        yaxis_title='Dividendos (R$)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def criar_grafico_dy(metricas_df):
    """Gráfico de Dividend Yield por ativo"""
    
    df_sorted = metricas_df.sort_values('dy_anual', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['ticker'],
        x=df_sorted['dy_anual'],
        orientation='h',
        marker_color='#2ecc71',
        hovertemplate='%{y}<br>DY: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Dividend Yield Anual',
        xaxis_title='DY (%)',
        yaxis_title='Ativo',
        height=max(400, len(df_sorted) * 30),
        showlegend=False
    )
    
    return fig


def criar_grafico_regularidade(metricas_df):
    """Gráfico de regularidade"""
    
    df_sorted = metricas_df.sort_values('regularidade', ascending=True)
    
    cores = ['#e74c3c' if r < 50 else '#f39c12' if r < 75 else '#2ecc71' 
             for r in df_sorted['regularidade']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['ticker'],
        x=df_sorted['regularidade'],
        orientation='h',
        marker_color=cores,
        hovertemplate='%{y}<br>Regularidade: %{x:.0f}/100<extra></extra>'
    ))
    
    fig.update_layout(
        title='Índice de Regularidade',
        xaxis_title='Regularidade (0-100)',
        yaxis_title='Ativo',
        height=max(400, len(df_sorted) * 30),
        showlegend=False
    )
    
    return fig


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal"""
    
    st.title("💰 Análise de Dividendos")
    st.markdown("Histórico, regularidade e calendário mensal simulado")
    st.markdown("---")
    
    # Verificar ativos
    if not st.session_state.portfolio_tickers:
        st.warning("⚠️ Nenhum ativo selecionado")
        st.info("👉 Vá para **Selecionar Ativos** primeiro")
        st.session_state.analise_dividendos_completa = False
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Cache info
        from core.cache import info_cache
        cache_info = info_cache()
        if cache_info['stats']['data_requests'] > 0:

            st.success(f"📦 {cache_info['entries']} cache(s)")
            if st.button("🗑️ Limpar Cache"):
                from core.cache import limpar_cache
                limpar_cache()
                st.session_state.analise_dividendos_completa = False
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("📅 Período")
        
        periodo_opcao = st.radio(
            "Selecione",
            ["1 ano", "2 anos", "5 anos", "Personalizado"],
            horizontal=True
        )
        
        if periodo_opcao == "Personalizado":
            col1, col2 = st.columns(2)
            with col1:
                data_inicio = st.date_input("Início", value=st.session_state.period_start)
            with col2:
                data_fim = st.date_input("Fim", value=st.session_state.period_end)
        else:
            anos = {"1 ano": 1, "2 anos": 2, "5 anos": 5}[periodo_opcao]
            data_fim = datetime.now()
            data_inicio = data_fim - timedelta(days=anos*365)
        
        # Atualizar
        novo_start = datetime.combine(data_inicio, datetime.min.time())
        novo_end = datetime.combine(data_fim, datetime.min.time())
        
        periodo_mudou = (
            novo_start != st.session_state.period_start or
            novo_end != st.session_state.period_end
        )
        
        st.session_state.period_start = novo_start
        st.session_state.period_end = novo_end
        
        if periodo_mudou:
            st.session_state.analise_dividendos_completa = False
        
        st.markdown("---")
        
        btn_analisar = st.button(
            "📊 Analisar Dividendos",
            type="primary",
            use_container_width=True
        )
    
    # Info
    st.info(f"📊 **{len(st.session_state.portfolio_tickers)} ativos** selecionados")
    
    with st.expander("📋 Ver lista"):
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            with cols[idx % 5]:
                st.write(f"• {ticker}")
    
    st.markdown("---")
    
    # Executar análise
    if btn_analisar:
        
        # Carregar dados
        with st.spinner("📥 Carregando dados..."):
            try:
                precos_df, dividendos_dict = carregar_dados_com_cache(
                    st.session_state.portfolio_tickers,
                    st.session_state.period_start,
                    st.session_state.period_end
                )
                
                if precos_df.empty:
                    st.error("❌ Erro ao carregar preços")
                    st.stop()
                
                if not dividendos_dict:
                    st.warning("⚠️ Nenhum dividendo encontrado")
                    st.stop()
                
                st.success(f"✅ {len(dividendos_dict)} ativos com dividendos")
                
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
                st.stop()
        
        # Calcular métricas
        with st.spinner("🧮 Calculando..."):
            metricas_lista = []
            calendario_carteira = {}
            
            for ticker, divs_df in dividendos_dict.items():
                # Preço médio
                if ticker in precos_df.columns:
                    preco_medio = precos_df[ticker].mean()
                else:
                    preco_medio = 0
                
                # Métricas
                dy = calcular_dividend_yield(divs_df, preco_medio)
                divs_mensais = agrupar_dividendos_por_mes(divs_df)
                regularidade = calcular_regularidade(divs_mensais)
                
                metricas_lista.append({
                    'ticker': ticker,
                    'dy_anual': dy,
                    'regularidade': regularidade,
                    'num_pagamentos': len(divs_df),
                    'total_dividendos': divs_df['valor'].sum(),
                    'preco_medio': preco_medio
                })
                
                # Calendário
                calendario_completo = criar_calendario_completo(
                    divs_mensais,
                    st.session_state.period_start,
                    st.session_state.period_end
                )
                calendario_carteira[ticker] = calendario_completo
            
            metricas_df = pd.DataFrame(metricas_lista)
            
            # Salvar
            st.session_state.metricas_dividendos = metricas_df
            st.session_state.calendario_dividendos = calendario_carteira
            st.session_state.analise_dividendos_completa = True
    
    # Exibir resultados
    if st.session_state.analise_dividendos_completa:
        
        metricas_df = st.session_state.metricas_dividendos
        calendario_carteira = st.session_state.calendario_dividendos
        
        st.header("📊 Visão Geral")
        
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ativos com Dividendos", len(metricas_df))
        
        with col2:
            st.metric("Total Pagamentos", int(metricas_df['num_pagamentos'].sum()))
        
        with col3:
            st.metric("DY Médio", f"{metricas_df['dy_anual'].mean():.2f}%")
        
        with col4:
            st.metric("Regularidade Média", f"{metricas_df['regularidade'].mean():.0f}/100")
        
        st.markdown("---")
        
        # Tabela
        st.subheader("📋 Métricas por Ativo")
        
        df_display = metricas_df.sort_values('dy_anual', ascending=False)
        
        st.dataframe(
            df_display.style.format({
                'dy_anual': '{:.2f}%',
                'regularidade': '{:.0f}',
                'total_dividendos': 'R$ {:.2f}',
                'preco_medio': 'R$ {:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Gráficos
        st.subheader("📈 Análise Comparativa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dy = criar_grafico_dy(metricas_df)
            if fig_dy:
                st.plotly_chart(fig_dy, use_container_width=True)
        
        with col2:
            fig_reg = criar_grafico_regularidade(metricas_df)
            if fig_reg:
                st.plotly_chart(fig_reg, use_container_width=True)
        
        st.markdown("---")
        
        # Calendário
        st.subheader("📅 Calendário Mensal")
        
        fig_heatmap = criar_heatmap_calendario(calendario_carteira)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Fluxo mensal
        st.subheader("💵 Fluxo Mensal Total")
        
        df_calendario = pd.DataFrame(calendario_carteira)
        dividendos_mensais_total = df_calendario.sum(axis=1)
        
        fig_mensal = criar_grafico_mensal(dividendos_mensais_total)
        if fig_mensal:
            st.plotly_chart(fig_mensal, use_container_width=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Média Mensal", f"R$ {dividendos_mensais_total.mean():.2f}")
        
        with col2:
            st.metric("Mediana", f"R$ {dividendos_mensais_total.median():.2f}")
        
        with col3:
            st.metric("Desvio Padrão", f"R$ {dividendos_mensais_total.std():.2f}")
        
        with col4:
            cobertura = ((dividendos_mensais_total > 0).sum() / len(dividendos_mensais_total)) * 100
            st.metric("Cobertura", f"{cobertura:.0f}%")
    
    else:
        st.info("👈 Configure o período e clique em **Analisar Dividendos**")


if __name__ == "__main__":
    main()
