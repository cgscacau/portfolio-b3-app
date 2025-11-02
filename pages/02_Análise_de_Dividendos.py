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

# Configuração da página
st.set_page_config(
    page_title="Análise de Dividendos",
    page_icon="💰",
    layout="wide"
)

# Inicializar
init_all()


# ==========================================
# FUNÇÕES DE CÁLCULO DE DIVIDENDOS
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
    if dividendos_df.empty or preco_medio == 0:
        return 0.0
    
    total_dividendos = dividendos_df['valor'].sum()
    return (total_dividendos / preco_medio) * 100


def calcular_regularidade(dividendos_mensais):
    """
    Calcula índice de regularidade dos dividendos
    Baseado no coeficiente de variação (CV = std/mean)
    Quanto menor, mais regular
    
    Args:
        dividendos_mensais: Series com dividendos por mês
        
    Returns:
        Float entre 0 e 100 (0 = irregular, 100 = muito regular)
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
    
    # Coeficiente de variação invertido e normalizado
    cv = std / media
    regularidade = max(0, 100 * (1 - min(cv, 1)))
    
    return regularidade


def agrupar_dividendos_por_mes(dividendos_df):
    """
    Agrupa dividendos por mês
    
    Args:
        dividendos_df: DataFrame com colunas ['data', 'valor']
        
    Returns:
        Series indexada por mês (YYYY-MM) com soma dos dividendos
    """
    if dividendos_df.empty:
        return pd.Series(dtype=float)
    
    df = dividendos_df.copy()
    df['mes'] = df['data'].dt.to_period('M')
    
    dividendos_mensais = df.groupby('mes')['valor'].sum()
    dividendos_mensais.index = dividendos_mensais.index.astype(str)
    
    return dividendos_mensais


def criar_calendario_completo(dividendos_mensais, data_inicio, data_fim):
    """
    Cria calendário completo preenchendo meses sem dividendos com zero
    
    Args:
        dividendos_mensais: Series com dividendos por mês
        data_inicio: Data de início
        data_fim: Data de fim
        
    Returns:
        Series com todos os meses do período
    """
    # Criar range de meses
    meses_completos = pd.period_range(
        start=data_inicio,
        end=data_fim,
        freq='M'
    )
    
    # Criar Series vazia
    calendario = pd.Series(0.0, index=meses_completos.astype(str))
    
    # Preencher com valores existentes
    for mes, valor in dividendos_mensais.items():
        if mes in calendario.index:
            calendario[mes] = valor
    
    return calendario


# ==========================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados_dividendos(tickers, data_inicio, data_fim):
    """
    Carrega dados de preços e dividendos para análise
    
    Args:
        tickers: Lista de tickers
        data_inicio: Data inicial
        data_fim: Data final
        
    Returns:
        Tuple (precos_df, dividendos_dict)
    """
    # Carregar preços
    precos_df = data.get_price_history(tickers, data_inicio, data_fim, use_cache=True)
    
    # Carregar dividendos
    dividendos_dict = {}
    
    for ticker in tickers:
        try:
            divs = data.get_dividends(ticker, data_inicio, data_fim)
            if not divs.empty:
                dividendos_dict[ticker] = divs
        except:
            continue
    
    return precos_df, dividendos_dict


# ==========================================
# FUNÇÕES DE VISUALIZAÇÃO
# ==========================================

def criar_heatmap_calendario(calendario_carteira, titulo="Calendário de Dividendos"):
    """
    Cria heatmap de dividendos mensais
    
    Args:
        calendario_carteira: Dict {ticker: Series mensal}
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    if not calendario_carteira:
        return None
    
    # Criar DataFrame para heatmap
    df_heatmap = pd.DataFrame(calendario_carteira)
    
    # Converter índice para datetime para ordenar
    df_heatmap.index = pd.to_datetime(df_heatmap.index + '-01')
    df_heatmap = df_heatmap.sort_index()
    
    # Formatar índice para exibição
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
        title=titulo,
        xaxis_title="Mês",
        yaxis_title="Ativo",
        height=max(400, len(df_heatmap.columns) * 30),
        hovermode='closest'
    )
    
    return fig


def criar_grafico_dividendos_mensais(dividendos_mensais_total, titulo="Dividendos Mensais da Carteira"):
    """
    Cria gráfico de barras dos dividendos mensais totais
    
    Args:
        dividendos_mensais_total: Series com dividendos por mês
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    if dividendos_mensais_total.empty:
        return None
    
    # Converter índice para datetime
    datas = pd.to_datetime(dividendos_mensais_total.index + '-01')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=datas,
        y=dividendos_mensais_total.values,
        marker_color='#3498db',
        hovertemplate='%{x|%B %Y}<br>R$ %{y:.2f}<extra></extra>'
    ))
    
    # Adicionar linha de média
    media = dividendos_mensais_total.mean()
    fig.add_hline(
        y=media,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Média: R$ {media:.2f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Mês",
        yaxis_title="Dividendos (R$)",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def criar_grafico_evolucao_dy(metricas_por_ativo):
    """
    Cria gráfico de barras com Dividend Yield por ativo
    
    Args:
        metricas_por_ativo: DataFrame com métricas
        
    Returns:
        Figura Plotly
    """
    df_sorted = metricas_por_ativo.sort_values('dy_anual', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['ticker'],
        x=df_sorted['dy_anual'],
        orientation='h',
        marker_color='#2ecc71',
        hovertemplate='%{y}<br>DY: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Dividend Yield Anual por Ativo",
        xaxis_title="DY (%)",
        yaxis_title="Ativo",
        height=max(400, len(df_sorted) * 30),
        showlegend=False
    )
    
    return fig


def criar_grafico_regularidade(metricas_por_ativo):
    """
    Cria gráfico de barras com índice de regularidade
    
    Args:
        metricas_por_ativo: DataFrame com métricas
        
    Returns:
        Figura Plotly
    """
    df_sorted = metricas_por_ativo.sort_values('regularidade', ascending=True)
    
    # Cores baseadas na regularidade
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
        title="Índice de Regularidade dos Dividendos",
        xaxis_title="Regularidade (0-100)",
        yaxis_title="Ativo",
        height=max(400, len(df_sorted) * 30),
        showlegend=False
    )
    
    return fig


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal da página"""
    
    st.title("💰 Análise de Dividendos")
    st.markdown("Histórico, regularidade e calendário mensal simulado de dividendos")
    st.markdown("---")
    
    # Verificar ativos selecionados
    if not st.session_state.portfolio_tickers:
        st.warning("⚠️ Nenhum ativo selecionado")
        st.info("👉 Vá para a página **Selecionar Ativos** para escolher os ativos")
        st.stop()
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.subheader("📅 Período de Análise")
        
        # Opções rápidas
        periodo_opcao = st.radio(
            "Período",
            ["1 ano", "2 anos", "5 anos", "Personalizado"],
            horizontal=True
        )
        
        if periodo_opcao == "Personalizado":
            col1, col2 = st.columns(2)
            with col1:
                data_inicio = st.date_input(
                    "Início",
                    value=st.session_state.period_start
                )
            with col2:
                data_fim = st.date_input(
                    "Fim",
                    value=st.session_state.period_end
                )
        else:
            anos = {"1 ano": 1, "2 anos": 2, "5 anos": 5}[periodo_opcao]
            data_fim = datetime.now()
            data_inicio = data_fim - timedelta(days=anos*365)
        
        # Atualizar session state
        st.session_state.period_start = datetime.combine(data_inicio, datetime.min.time())
        st.session_state.period_end = datetime.combine(data_fim, datetime.min.time())
        
        st.markdown("---")
        
        # Botão de análise
        btn_analisar = st.button(
            "📊 Analisar Dividendos",
            type="primary",
            use_container_width=True
        )
    
    # Informações dos ativos
    st.info(f"📊 **{len(st.session_state.portfolio_tickers)} ativos** selecionados para análise")
    
    with st.expander("📋 Ver lista de ativos"):
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            with cols[idx % 5]:
                st.write(f"• {ticker}")
    
    st.markdown("---")
    
    # Executar análise
    if btn_analisar:
        
        # Carregar dados
        with st.spinner("📥 Carregando dados de preços e dividendos..."):
            try:
                precos_df, dividendos_dict = carregar_dados_dividendos(
                    st.session_state.portfolio_tickers,
                    st.session_state.period_start,
                    st.session_state.period_end
                )
                
                if precos_df.empty:
                    st.error("❌ Não foi possível carregar dados de preços")
                    st.stop()
                
                if not dividendos_dict:
                    st.warning("⚠️ Nenhum dividendo encontrado no período")
                    st.stop()
                
                st.success(f"✅ Dados carregados: **{len(dividendos_dict)} ativos** com dividendos")
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar dados: {str(e)}")
                st.stop()
        
        # Calcular métricas por ativo
        with st.spinner("🧮 Calculando métricas..."):
            metricas_lista = []
            calendario_carteira = {}
            
            for ticker, divs_df in dividendos_dict.items():
                # Preço médio do período
                if ticker in precos_df.columns:
                    preco_medio = precos_df[ticker].mean()
                else:
                    preco_medio = 0
                
                # Dividend Yield
                dy = calcular_dividend_yield(divs_df, preco_medio)
                
                # Dividendos mensais
                divs_mensais = agrupar_dividendos_por_mes(divs_df)
                
                # Regularidade
                regularidade = calcular_regularidade(divs_mensais)
                
                # Número de pagamentos
                num_pagamentos = len(divs_df)
                
                # Total de dividendos
                total_divs = divs_df['valor'].sum()
                
                metricas_lista.append({
                    'ticker': ticker,
                    'dy_anual': dy,
                    'regularidade': regularidade,
                    'num_pagamentos': num_pagamentos,
                    'total_dividendos': total_divs,
                    'preco_medio': preco_medio
                })
                
                # Calendário completo
                calendario_completo = criar_calendario_completo(
                    divs_mensais,
                    st.session_state.period_start,
                    st.session_state.period_end
                )
                calendario_carteira[ticker] = calendario_completo
            
            metricas_df = pd.DataFrame(metricas_lista)
        
        # ==========================================
        # EXIBIR RESULTADOS
        # ==========================================
        
        st.header("📊 Visão Geral")
        
        # Métricas resumidas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Ativos com Dividendos",
                len(dividendos_dict),
                help="Número de ativos que pagaram dividendos no período"
            )
        
        with col2:
            total_pagamentos = metricas_df['num_pagamentos'].sum()
            st.metric(
                "Total de Pagamentos",
                f"{total_pagamentos}",
                help="Soma de todos os eventos de pagamento"
            )
        
        with col3:
            dy_medio = metricas_df['dy_anual'].mean()
            st.metric(
                "DY Médio",
                f"{dy_medio:.2f}%",
                help="Dividend Yield médio dos ativos"
            )
        
        with col4:
            reg_media = metricas_df['regularidade'].mean()
            st.metric(
                "Regularidade Média",
                f"{reg_media:.0f}/100",
                help="Índice médio de regularidade (0-100)"
            )
        
        st.markdown("---")
        
        # Tabela detalhada
        st.subheader("📋 Métricas Detalhadas por Ativo")
        
        df_display = metricas_df.copy()
        df_display = df_display.sort_values('dy_anual', ascending=False)
        
        st.dataframe(
            df_display.style.format({
                'dy_anual': '{:.2f}%',
                'regularidade': '{:.0f}',
                'total_dividendos': 'R$ {:.2f}',
                'preco_medio': 'R$ {:.2f}'
            }),
            column_config={
                'ticker': st.column_config.TextColumn('Ativo', width='small'),
                'dy_anual': st.column_config.NumberColumn('DY Anual', width='small'),
                'regularidade': st.column_config.NumberColumn('Regularidade', width='small'),
                'num_pagamentos': st.column_config.NumberColumn('Pagamentos', width='small'),
                'total_dividendos': st.column_config.NumberColumn('Total Dividendos', width='medium'),
                'preco_medio': st.column_config.NumberColumn('Preço Médio', width='medium')
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Gráficos de DY e Regularidade
        st.subheader("📈 Análise Comparativa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dy = criar_grafico_evolucao_dy(metricas_df)
            if fig_dy:
                st.plotly_chart(fig_dy, use_container_width=True)
        
        with col2:
            fig_reg = criar_grafico_regularidade(metricas_df)
            if fig_reg:
                st.plotly_chart(fig_reg, use_container_width=True)
        
        st.markdown("---")
        
        # Calendário mensal
        st.subheader("📅 Calendário Mensal de Dividendos")
        
        # Heatmap
        fig_heatmap = criar_heatmap_calendario(calendario_carteira)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Dividendos mensais totais da carteira
        st.subheader("💵 Fluxo Mensal Total da Carteira")
        
        # Somar todos os dividendos por mês
        df_calendario = pd.DataFrame(calendario_carteira)
        dividendos_mensais_total = df_calendario.sum(axis=1)
        
        fig_mensal = criar_grafico_dividendos_mensais(dividendos_mensais_total)
        if fig_mensal:
            st.plotly_chart(fig_mensal, use_container_width=True)
        
        # Estatísticas do fluxo mensal
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            media_mensal = dividendos_mensais_total.mean()
            st.metric("Média Mensal", f"R$ {media_mensal:.2f}")
        
        with col2:
            mediana_mensal = dividendos_mensais_total.median()
            st.metric("Mediana Mensal", f"R$ {mediana_mensal:.2f}")
        
        with col3:
            std_mensal = dividendos_mensais_total.std()
            st.metric("Desvio Padrão", f"R$ {std_mensal:.2f}")
        
        with col4:
            meses_com_divs = (dividendos_mensais_total > 0).sum()
            total_meses = len(dividendos_mensais_total)
            cobertura = (meses_com_divs / total_meses) * 100
            st.metric("Cobertura", f"{cobertura:.0f}%", 
                     help="Percentual de meses com dividendos")
        
        st.markdown("---")
        
        # Informações e dicas
        with st.expander("ℹ️ Como interpretar os resultados"):
            st.markdown("""
            ### 📊 Dividend Yield (DY)
            - Percentual de retorno em dividendos em relação ao preço médio
            - **DY > 6%**: Considerado bom para ações brasileiras
            - **DY > 8%**: Excelente para FIIs
            
            ### 📈 Índice de Regularidade
            - Mede a consistência dos pagamentos mensais
            - **0-50**: Irregular (pagamentos esporádicos)
            - **50-75**: Moderado (alguma previsibilidade)
            - **75-100**: Regular (fluxo consistente)
            
            ### 📅 Calendário Mensal
            - Visualiza quando cada ativo paga dividendos
            - Permite identificar concentração de pagamentos
            - Ideal: distribuição uniforme ao longo dos meses
            
            ### 💡 Dicas
            - Combine ativos com diferentes meses de pagamento
            - Priorize regularidade para renda mensal estável
            - DY muito alto pode indicar risco (verifique fundamentals)
            """)
    
    else:
        # Mensagem inicial
        st.info("👈 Configure o período na barra lateral e clique em **Analisar Dividendos**")
        
        # Informações sobre a análise
        with st.expander("ℹ️ Sobre esta análise"):
            st.markdown("""
            ## 💰 Análise de Dividendos
            
            Esta página oferece uma análise completa dos dividendos pagos pelos ativos selecionados:
            
            ### 📊 Métricas Calculadas
            
            1. **Dividend Yield (DY)**: Retorno percentual em dividendos
            2. **Regularidade**: Consistência dos pagamentos ao longo do tempo
            3. **Número de Pagamentos**: Frequência de distribuição
            4. **Calendário Mensal**: Visualização temporal dos pagamentos
            
            ### 🎯 Objetivos
            
            - Identificar ativos com bom retorno em dividendos
            - Avaliar a previsibilidade dos fluxos de caixa
            - Planejar uma carteira com renda mensal estável
            - Visualizar a distribuição temporal dos pagamentos
            
            ### 📈 Como usar
            
            1. Selecione o período de análise (1, 2, 5 anos ou personalizado)
            2. Clique em "Analisar Dividendos"
            3. Analise as métricas e gráficos
            4. Use as informações para construir sua estratégia de renda
            """)


if __name__ == "__main__":
    main()
