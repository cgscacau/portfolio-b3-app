"""
Resumo Executivo - Dashboard Consolidado
Visão geral completa de todos os portfólios e análises
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importar módulos
from core.data import get_price_history, get_dividends, obter_preco_atual
from core.cache import cache_manager
from core.portfolio import portfolio_manager, obter_portfolio_ativo, listar_portfolios


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Resumo Executivo",
    page_icon="📊",
    layout="wide"
)

# Painel de cache (opcional)
try:
    cache_manager.exibir_painel_controle()
except AttributeError:
    pass  # Método não disponível, continua sem o painel


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def calcular_metricas_portfolio(df_precos, pesos):
    """Calcula métricas de performance do portfólio"""
    if df_precos.empty:
        return None
    
    # Retornos
    df_retornos = df_precos.pct_change().dropna()
    
    # Retorno do portfólio
    retorno_portfolio = (df_retornos * pesos).sum(axis=1)
    
    # Métricas
    retorno_acumulado = (1 + retorno_portfolio).cumprod()
    retorno_total = (retorno_acumulado.iloc[-1] - 1) * 100
    
    # Anualizar
    dias = len(retorno_portfolio)
    anos = dias / 252
    retorno_anual = ((1 + retorno_total/100) ** (1/anos) - 1) * 100 if anos > 0 else 0
    
    volatilidade = retorno_portfolio.std() * (252 ** 0.5) * 100
    sharpe = (retorno_anual / volatilidade) if volatilidade > 0 else 0
    
    # Drawdown
    cumulative = retorno_acumulado
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    return {
        'retorno_total': retorno_total,
        'retorno_anual': retorno_anual,
        'volatilidade': volatilidade,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'retorno_acumulado': retorno_acumulado,
        'drawdown': drawdown
    }


def buscar_dividendos_portfolio(tickers, data_inicio, data_fim):
    """Busca dividendos de todos os ativos do portfólio"""
    dividendos_total = []
    
    for ticker in tickers:
        df_div = get_dividends(ticker, data_inicio, data_fim)
        if not df_div.empty:
            df_div['ticker'] = ticker
            dividendos_total.append(df_div)
    
    if dividendos_total:
        return pd.concat(dividendos_total, ignore_index=True)
    
    return pd.DataFrame(columns=['data', 'valor', 'ticker'])


# ==========================================
# TÍTULO E INTRODUÇÃO
# ==========================================

st.title("📊 Resumo Executivo")
st.markdown("Dashboard consolidado com visão geral de todos os seus investimentos")

# Data e hora da atualização
st.caption(f"📅 Atualizado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")

st.markdown("---")


# ==========================================
# SELEÇÃO DE PORTFÓLIO OU CONFIGURAÇÃO RÁPIDA
# ==========================================

portfolios_disponiveis = listar_portfolios()
portfolio_ativo = obter_portfolio_ativo()

col1, col2 = st.columns([3, 1])

with col1:
    if portfolios_disponiveis:
        # Usar portfólio salvo
        usar_portfolio = st.checkbox(
            "Usar portfólio salvo",
            value=True if portfolio_ativo else False
        )
        
        if usar_portfolio:
            portfolio_selecionado = st.selectbox(
                "Selecione o portfólio",
                portfolios_disponiveis,
                index=portfolios_disponiveis.index(portfolio_ativo.nome) if portfolio_ativo else 0
            )
            
            portfolio = portfolio_manager.carregar(portfolio_selecionado)
            
            if portfolio:
                tickers = portfolio.tickers
                pesos = portfolio.pesos
                data_inicio = portfolio.data_inicio
                data_fim = portfolio.data_fim
                st.success(f"✅ Portfólio '{portfolio.nome}' carregado")
        else:
            usar_portfolio = False
    else:
        usar_portfolio = False
        st.info("💡 Nenhum portfólio salvo. Use configuração rápida abaixo.")

# Configuração rápida se não usar portfólio
if not usar_portfolio or not portfolios_disponiveis:
    st.markdown("### ⚡ Configuração Rápida")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickers_input = st.text_input(
            "Ativos (separados por vírgula)",
            value="PETR4,VALE3,ITUB4",
            help="Ex: PETR4,VALE3,ITUB4"
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    with col2:
        data_fim = st.date_input(
            "Data Final",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    with col3:
        periodo = st.selectbox(
            "Período",
            ["1 mês", "3 meses", "6 meses", "1 ano", "2 anos", "5 anos"],
            index=3
        )
        
        periodos_dias = {
            "1 mês": 30,
            "3 meses": 90,
            "6 meses": 180,
            "1 ano": 365,
            "2 anos": 730,
            "5 anos": 1825
        }
        
        data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=periodos_dias[periodo])
        data_fim = datetime.combine(data_fim, datetime.min.time())
    
    # Pesos iguais
    pesos = [1.0 / len(tickers)] * len(tickers)
    
    st.info(f"📊 Analisando {len(tickers)} ativos com pesos iguais ({100/len(tickers):.1f}% cada)")

with col2:
    if st.button("🔄 Atualizar Dados", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ==========================================
# VALIDAÇÃO
# ==========================================

if not tickers:
    st.error("❌ Nenhum ativo selecionado!")
    st.stop()

st.markdown("---")


# ==========================================
# BUSCAR DADOS
# ==========================================

with st.spinner("📡 Buscando dados do mercado..."):
    # Preços históricos
    df_precos = get_price_history(tickers, data_inicio, data_fim)
    
    # Dividendos
    df_dividendos = buscar_dividendos_portfolio(tickers, data_inicio, data_fim)
    
    # Preços atuais
    precos_atuais = {ticker: obter_preco_atual(ticker) for ticker in tickers}


# Validar dados
if df_precos.empty:
    st.error("❌ Não foi possível obter dados de preços!")
    st.stop()


# ==========================================
# CALCULAR MÉTRICAS
# ==========================================

metricas = calcular_metricas_portfolio(df_precos, pesos)

if not metricas:
    st.error("❌ Erro ao calcular métricas!")
    st.stop()


# ==========================================
# SEÇÃO 1: VISÃO GERAL - KPIs PRINCIPAIS
# ==========================================

st.header("📈 Visão Geral")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Retorno Total",
        f"{metricas['retorno_total']:.2f}%",
        delta=f"{metricas['retorno_anual']:.2f}% a.a."
    )

with col2:
    st.metric(
        "Volatilidade",
        f"{metricas['volatilidade']:.2f}%",
        delta="Anual"
    )

with col3:
    cor_sharpe = "normal" if metricas['sharpe'] > 1 else "inverse"
    st.metric(
        "Sharpe Ratio",
        f"{metricas['sharpe']:.2f}",
        delta="Risco/Retorno",
        delta_color=cor_sharpe
    )

with col4:
    st.metric(
        "Max Drawdown",
        f"{metricas['max_drawdown']:.2f}%",
        delta="Maior queda"
    )

with col5:
    total_dividendos = df_dividendos['valor'].sum() if not df_dividendos.empty else 0
    st.metric(
        "Dividendos",
        f"R$ {total_dividendos:.2f}",
        delta=f"{len(df_dividendos)} pagamentos"
    )

st.markdown("---")


# ==========================================
# SEÇÃO 2: EVOLUÇÃO DO PORTFÓLIO
# ==========================================

st.header("📊 Evolução do Portfólio")

# Criar gráfico com 2 subplots
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    subplot_titles=("Retorno Acumulado", "Drawdown"),
    vertical_spacing=0.1
)

# Subplot 1: Retorno Acumulado
fig.add_trace(
    go.Scatter(
        x=metricas['retorno_acumulado'].index,
        y=metricas['retorno_acumulado'].values,
        mode='lines',
        name='Portfólio',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ),
    row=1, col=1
)

# Linha de referência (1.0)
fig.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="gray",
    opacity=0.5,
    row=1, col=1
)

# Subplot 2: Drawdown
fig.add_trace(
    go.Scatter(
        x=metricas['drawdown'].index,
        y=metricas['drawdown'].values,
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ),
    row=2, col=1
)

# Layout
fig.update_xaxes(title_text="Data", row=2, col=1)
fig.update_yaxes(title_text="Valor Acumulado", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

fig.update_layout(
    height=600,
    showlegend=True,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ==========================================
# SEÇÃO 3: COMPOSIÇÃO DO PORTFÓLIO
# ==========================================

st.header("🎯 Composição do Portfólio")

col1, col2 = st.columns([1, 2])

with col1:
    # Gráfico de pizza
    fig_pizza = go.Figure(data=[go.Pie(
        labels=tickers,
        values=[p*100 for p in pesos],
        hole=0.4,
        marker=dict(line=dict(color='white', width=2))
    )])
    
    fig_pizza.update_layout(
        title="Alocação por Ativo",
        height=400
    )
    
    st.plotly_chart(fig_pizza, use_container_width=True)

with col2:
    # Tabela de composição com métricas
    dados_composicao = []
    
    for i, ticker in enumerate(tickers):
        preco_atual = precos_atuais.get(ticker, 0)
        
        if ticker in df_precos.columns:
            preco_inicial = df_precos[ticker].iloc[0]
            preco_final = df_precos[ticker].iloc[-1]
            retorno = ((preco_final / preco_inicial) - 1) * 100 if preco_inicial > 0 else 0
            
            # Dividendos deste ativo
            div_ativo = df_dividendos[df_dividendos['ticker'] == ticker]['valor'].sum() if not df_dividendos.empty else 0
        else:
            retorno = 0
            div_ativo = 0
        
        dados_composicao.append({
            'Ativo': ticker,
            'Peso': f"{pesos[i]*100:.1f}%",
            'Preço Atual': f"R$ {preco_atual:.2f}" if preco_atual else "N/A",
            'Retorno': f"{retorno:+.2f}%",
            'Dividendos': f"R$ {div_ativo:.2f}"
        })
    
    df_composicao = pd.DataFrame(dados_composicao)
    
    st.dataframe(
        df_composicao,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Ativo': st.column_config.TextColumn('Ativo', width="small"),
            'Peso': st.column_config.TextColumn('Peso', width="small"),
            'Preço Atual': st.column_config.TextColumn('Preço Atual', width="medium"),
            'Retorno': st.column_config.TextColumn('Retorno', width="small"),
            'Dividendos': st.column_config.TextColumn('Dividendos', width="small")
        }
    )

st.markdown("---")


# ==========================================
# SEÇÃO 4: ANÁLISE DE DIVIDENDOS
# ==========================================

st.header("💰 Análise de Dividendos")

if df_dividendos.empty:
    st.info("📭 Nenhum dividendo registrado no período analisado")
else:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Recebido", f"R$ {df_dividendos['valor'].sum():.2f}")
    
    with col2:
        st.metric("Número de Pagamentos", len(df_dividendos))
    
    with col3:
        media_pagamento = df_dividendos['valor'].mean()
        st.metric("Média por Pagamento", f"R$ {media_pagamento:.2f}")
    
    # Gráfico de dividendos ao longo do tempo
    fig_div = go.Figure()
    
    # Agrupar por mês
    df_dividendos['mes'] = pd.to_datetime(df_dividendos['data']).dt.to_period('M')
    div_mensal = df_dividendos.groupby('mes')['valor'].sum().reset_index()
    div_mensal['mes'] = div_mensal['mes'].astype(str)
    
    fig_div.add_trace(go.Bar(
        x=div_mensal['mes'],
        y=div_mensal['valor'],
        name='Dividendos Mensais',
        marker_color='green'
    ))
    
    fig_div.update_layout(
        title="Dividendos Recebidos por Mês",
        xaxis_title="Mês",
        yaxis_title="Valor (R$)",
        height=400
    )
    
    st.plotly_chart(fig_div, use_container_width=True)
    
    # Tabela de dividendos por ativo
    st.markdown("**Dividendos por Ativo:**")
    div_por_ativo = df_dividendos.groupby('ticker')['valor'].agg(['sum', 'count', 'mean']).reset_index()
    div_por_ativo.columns = ['Ativo', 'Total (R$)', 'Pagamentos', 'Média (R$)']
    div_por_ativo = div_por_ativo.sort_values('Total (R$)', ascending=False)
    
    st.dataframe(
        div_por_ativo,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Total (R$)': st.column_config.NumberColumn('Total (R$)', format="R$ %.2f"),
            'Média (R$)': st.column_config.NumberColumn('Média (R$)', format="R$ %.2f")
        }
    )

st.markdown("---")


# ==========================================
# SEÇÃO 5: COMPARAÇÃO DE ATIVOS
# ==========================================

st.header("⚖️ Comparação de Ativos")

# Normalizar preços (base 100)
df_normalizado = df_precos / df_precos.iloc[0] * 100

# Gráfico de linhas
fig_comp = go.Figure()

for ticker in tickers:
    if ticker in df_normalizado.columns:
        fig_comp.add_trace(go.Scatter(
            x=df_normalizado.index,
            y=df_normalizado[ticker],
            mode='lines',
            name=ticker
        ))

fig_comp.update_layout(
    title="Performance Relativa dos Ativos (Base 100)",
    xaxis_title="Data",
    yaxis_title="Valor (Base 100)",
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig_comp, use_container_width=True)

# Tabela de performance
st.markdown("**Ranking de Performance:**")

performance_ativos = []
for ticker in tickers:
    if ticker in df_precos.columns:
        retorno = ((df_precos[ticker].iloc[-1] / df_precos[ticker].iloc[0]) - 1) * 100
        volatilidade = df_precos[ticker].pct_change().std() * (252 ** 0.5) * 100
        sharpe_ativo = (retorno / volatilidade) if volatilidade > 0 else 0
        
        performance_ativos.append({
            'Ativo': ticker,
            'Retorno': retorno,
            'Volatilidade': volatilidade,
            'Sharpe': sharpe_ativo
        })

df_performance = pd.DataFrame(performance_ativos)
df_performance = df_performance.sort_values('Retorno', ascending=False)

st.dataframe(
    df_performance,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Retorno': st.column_config.NumberColumn('Retorno (%)', format="%.2f%%"),
        'Volatilidade': st.column_config.NumberColumn('Volatilidade (%)', format="%.2f%%"),
        'Sharpe': st.column_config.NumberColumn('Sharpe', format="%.2f")
    }
)

st.markdown("---")


# ==========================================
# SEÇÃO 6: RECOMENDAÇÕES
# ==========================================

st.header("💡 Recomendações")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Pontos Positivos")
    
    pontos_positivos = []
    
    if metricas['retorno_anual'] > 10:
        pontos_positivos.append("✅ Retorno anualizado acima de 10%")
    
    if metricas['sharpe'] > 1:
        pontos_positivos.append("✅ Sharpe Ratio acima de 1 (boa relação risco/retorno)")
    
    if metricas['volatilidade'] < 30:
        pontos_positivos.append("✅ Volatilidade controlada (abaixo de 30%)")
    
    if not df_dividendos.empty:
        pontos_positivos.append(f"✅ Recebendo dividendos regularmente ({len(df_dividendos)} pagamentos)")
    
    if len(tickers) >= 5:
        pontos_positivos.append("✅ Boa diversificação (5+ ativos)")
    
    if pontos_positivos:
        for ponto in pontos_positivos:
            st.success(ponto)
    else:
        st.info("Nenhum ponto positivo identificado no momento")

with col2:
    st.markdown("### ⚠️ Pontos de Atenção")
    
    pontos_atencao = []
    
    if metricas['retorno_anual'] < 0:
        pontos_atencao.append("⚠️ Retorno anualizado negativo")
    
    if metricas['sharpe'] < 0.5:
        pontos_atencao.append("⚠️ Sharpe Ratio baixo (relação risco/retorno ruim)")
    
    if metricas['volatilidade'] > 40:
        pontos_atencao.append("⚠️ Alta volatilidade (acima de 40%)")
    
    if abs(metricas['max_drawdown']) > 30:
        pontos_atencao.append(f"⚠️ Drawdown elevado ({metricas['max_drawdown']:.1f}%)")
    
    if len(tickers) < 3:
        pontos_atencao.append("⚠️ Portfólio pouco diversificado (menos de 3 ativos)")
    
    # Verificar concentração
    max_peso = max(pesos) * 100
    if max_peso > 50:
        pontos_atencao.append(f"⚠️ Alta concentração em um ativo ({max_peso:.1f}%)")
    
    if pontos_atencao:
        for ponto in pontos_atencao:
            st.warning(ponto)
    else:
        st.success("Nenhum ponto de atenção identificado! 🎉")


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📅 Período Analisado:**")
    st.write(f"{data_inicio.date()} até {data_fim.date()}")
    st.write(f"{(data_fim - data_inicio).days} dias úteis")

with col2:
    st.markdown("**📊 Dados:**")
    st.write(f"{len(tickers)} ativos analisados")
    st.write(f"{len(df_precos)} dias de cotações")

with col3:
    st.markdown("**⚡ Cache:**")
    cache_info = cache_manager.obter_info()
    st.write(f"Taxa de acerto: {cache_info['stats']['hit_rate']:.1f}%")
    st.write(f"{cache_info['stats']['data_requests']} requisições")

st.markdown("---")
st.caption("💡 **Dica:** Este resumo é atualizado automaticamente. Use o botão 'Atualizar Dados' para forçar nova busca.")
