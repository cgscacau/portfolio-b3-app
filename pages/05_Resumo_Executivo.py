"""
📋 Resumo Executivo
Recomendação final personalizada com plano de investimento e projeção realista de dividendos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import sys
from pathlib import Path

# Configurar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core.init import init_all
from core.cache import carregar_dados_cache
from core import data

# Configuração da página
st.set_page_config(
    page_title="Resumo Executivo",
    page_icon="📋",
    layout="wide"
)

# Inicializar
init_all()


# ==========================================
# FUNÇÕES DE OTIMIZAÇÃO
# ==========================================

def calcular_retornos(prices):
    """Calcula retornos diários"""
    return prices.pct_change().dropna()


def portfolio_return(weights, returns):
    """Calcula retorno anualizado do portfólio"""
    return np.sum(returns.mean() * weights) * 252


def portfolio_volatility(weights, returns):
    """Calcula volatilidade anualizada do portfólio"""
    cov_matrix = returns.cov() * 252
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)


def sharpe_ratio(weights, returns, rf_rate):
    """Calcula Sharpe Ratio"""
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, returns)
    return (ret - rf_rate) / vol if vol > 0 else 0


def otimizar_sharpe_maximo(returns, rf_rate):
    """Otimiza para máximo Sharpe"""
    n_assets = len(returns.columns)
    
    def objetivo(weights):
        return -sharpe_ratio(weights, returns, rf_rate)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(objetivo, initial, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    
    if not result.success:
        return None
    
    weights = result.x
    
    return {
        'weights': dict(zip(returns.columns, weights)),
        'return': portfolio_return(weights, returns),
        'volatility': portfolio_volatility(weights, returns),
        'sharpe': sharpe_ratio(weights, returns, rf_rate)
    }


def otimizar_minima_volatilidade(returns):
    """Otimiza para mínima volatilidade"""
    n_assets = len(returns.columns)
    
    def objetivo(weights):
        return portfolio_volatility(weights, returns)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(objetivo, initial, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    
    if not result.success:
        return None
    
    weights = result.x
    
    return {
        'weights': dict(zip(returns.columns, weights)),
        'return': portfolio_return(weights, returns),
        'volatility': portfolio_volatility(weights, returns),
        'sharpe': sharpe_ratio(weights, returns, 0.0)
    }


def otimizar_dividendos(returns, metricas_dividendos):
    """
    Otimiza para dividendos regulares
    Prioriza ativos com alto DY e boa regularidade
    """
    if metricas_dividendos is None or metricas_dividendos.empty:
        return otimizar_sharpe_maximo(returns, 0.1175)
    
    n_assets = len(returns.columns)
    
    # Criar score de dividendos para cada ativo
    scores = {}
    for _, row in metricas_dividendos.iterrows():
        ticker = row['ticker']
        if ticker in returns.columns:
            # Score = DY + Regularidade/100
            score = row['dy_anual'] + (row['regularidade'] / 100) * 5
            scores[ticker] = score
    
    # Função objetivo: maximizar score de dividendos
    def objetivo(weights):
        score_total = 0
        for i, ticker in enumerate(returns.columns):
            score_total += weights[i] * scores.get(ticker, 0)
        return -score_total
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 0.25) for _ in range(n_assets))  # Max 25% por ativo
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(objetivo, initial, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    
    if not result.success:
        return None
    
    weights = result.x
    
    return {
        'weights': dict(zip(returns.columns, weights)),
        'return': portfolio_return(weights, returns),
        'volatility': portfolio_volatility(weights, returns),
        'sharpe': sharpe_ratio(weights, returns, 0.1175)
    }


# ==========================================
# RECOMENDAÇÃO
# ==========================================

def gerar_recomendacao_por_objetivo(objetivo_usuario, portfolios, perfil):
    """
    Gera recomendação baseada no objetivo escolhido
    
    Args:
        objetivo_usuario: String com objetivo
        portfolios: Dict com portfólios
        perfil: Perfil do investidor
        
    Returns:
        Dict com recomendação
    """
    # Mapear objetivo para portfólio
    if objetivo_usuario == "Máximo retorno ajustado ao risco":
        portfolio_escolhido = 'sharpe_maximo'
        
    elif objetivo_usuario == "Mínima volatilidade":
        portfolio_escolhido = 'minima_volatilidade'
        
    elif objetivo_usuario == "Renda mensal de dividendos":
        portfolio_escolhido = 'dividendos_regulares'
        
    else:  # "Deixar o sistema decidir"
        # Decidir baseado no perfil
        if perfil == 'conservador':
            portfolio_escolhido = 'minima_volatilidade'
        elif perfil == 'agressivo':
            portfolio_escolhido = 'sharpe_maximo'
        else:  # moderado
            # Escolher melhor Sharpe
            melhor = max(portfolios.items(), key=lambda x: x[1]['sharpe'])
            portfolio_escolhido = melhor[0]
    
    # Verificar se existe
    if portfolio_escolhido not in portfolios:
        portfolio_escolhido = list(portfolios.keys())[0]
    
    # Informações
    explicacoes = {
        'sharpe_maximo': {
            'titulo': '🎯 Portfólio de Máximo Sharpe Ratio',
            'descricao': 'Melhor relação risco-retorno disponível',
            'indicado': 'Investidores que buscam eficiência e crescimento balanceado',
            'vantagens': [
                'Otimiza retorno ajustado ao risco',
                'Equilíbrio entre ganhos e volatilidade',
                'Estratégia comprovada pela teoria moderna',
                'Ideal para perfil moderado a agressivo'
            ],
            'emoji': '🎯'
        },
        'minima_volatilidade': {
            'titulo': '🛡️ Portfólio de Mínima Volatilidade',
            'descricao': 'Máxima estabilidade e menor risco possível',
            'indicado': 'Investidores conservadores que priorizam preservação de capital',
            'vantagens': [
                'Menor oscilação de preços',
                'Ideal para perfil conservador',
                'Proteção em momentos de crise',
                'Maior previsibilidade de resultados'
            ],
            'emoji': '🛡️'
        },
        'dividendos_regulares': {
            'titulo': '💰 Portfólio de Dividendos Regulares',
            'descricao': 'Foco em renda passiva mensal consistente',
            'indicado': 'Investidores que buscam fluxo de caixa regular',
            'vantagens': [
                'Renda mensal previsível',
                'Bons pagadores históricos',
                'Estratégia de longo prazo',
                'Ideal para complemento de renda'
            ],
            'emoji': '💰'
        }
    }
    
    return {
        'portfolio': portfolio_escolhido,
        'dados': portfolios[portfolio_escolhido],
        'info': explicacoes[portfolio_escolhido],
        'motivo': objetivo_usuario
    }


def analisar_perfil_investidor(valor):
    """Determina perfil baseado no valor"""
    if valor < 10000:
        return 'conservador'
    elif valor < 50000:
        return 'moderado'
    else:
        return 'agressivo'


def calcular_quantidades(weights, valor_investimento, precos_atuais):
    """Calcula quantidades a comprar"""
    alocacoes = []
    
    for ticker, peso in weights.items():
        if peso < 0.01:
            continue
        
        valor_alocar = valor_investimento * peso
        preco = precos_atuais.get(ticker, 0)
        
        if preco > 0:
            quantidade = int(valor_alocar / preco)
            valor_real = quantidade * preco
            
            alocacoes.append({
                'Ativo': ticker,
                'Peso': peso,
                'Valor Alvo': valor_alocar,
                'Preço': preco,
                'Quantidade': quantidade,
                'Valor Real': valor_real
            })
    
    df = pd.DataFrame(alocacoes)
    
    if not df.empty:
        df = df.sort_values('Valor Real', ascending=False)
    
    return df


# ==========================================
# PROJEÇÃO DE DIVIDENDOS - CORRIGIDA
# ==========================================

def projetar_dividendos_futuros(tickers, weights, valor_investimento, precos_atuais, meses=12):
    """
    Projeta dividendos mensais baseado no PADRÃO HISTÓRICO REAL
    Respeita os meses específicos em que cada ativo costuma pagar
    
    Args:
        tickers: Lista de tickers
        weights: Dict com pesos
        valor_investimento: Valor total
        precos_atuais: Dict com preços
        meses: Meses para projetar
        
    Returns:
        DataFrame com projeção mensal realista
    """
    # Buscar histórico de 3 anos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)
    
    dividendos_historicos = {}
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status.text(f"Analisando histórico de {ticker}...")
        try:
            divs = data.get_dividends(ticker, start_date, end_date)
            if not divs.empty:
                dividendos_historicos[ticker] = divs
        except:
            continue
        progress.progress((idx + 1) / len(tickers))
    
    progress.empty()
    status.empty()
    
    if not dividendos_historicos:
        return pd.DataFrame()
    
    # Analisar padrão de pagamento de cada ativo
    padroes_pagamento = {}
    
    for ticker, divs_df in dividendos_historicos.items():
        divs_df = divs_df.copy()
        divs_df['data'] = pd.to_datetime(divs_df['data'])
        divs_df['mes_numero'] = divs_df['data'].dt.month
        divs_df['ano'] = divs_df['data'].dt.year
        
        # Calcular valor médio POR MÊS quando paga
        valores_por_mes = divs_df.groupby('mes_numero')['valor'].mean().to_dict()
        
        # Identificar quais meses costuma pagar
        meses_pagamento = divs_df['mes_numero'].unique().tolist()
        
        padroes_pagamento[ticker] = {
            'meses_pagamento': meses_pagamento,
            'valores_medios': valores_por_mes
        }
    
    # Calcular quantidades
    quantidades = {}
    
    for ticker, peso in weights.items():
        if peso < 0.01:
            continue
        
        valor_alocar = valor_investimento * peso
        preco = precos_atuais.get(ticker, 0)
        
        if preco > 0:
            quantidade = int(valor_alocar / preco)
            quantidades[ticker] = quantidade
    
    # Projetar mês a mês
    projecao = []
    data_inicio = datetime.now()
    
    for i in range(meses):
        mes_data = data_inicio + timedelta(days=30 * i)
        mes_numero = mes_data.month
        mes_nome = mes_data.strftime('%b/%Y')
        
        dividendo_total = 0
        detalhes = {}
        
        for ticker, quantidade in quantidades.items():
            if ticker not in padroes_pagamento:
                continue
            
            padrao = padroes_pagamento[ticker]
            
            # Verificar se o ativo PAGA neste mês
            if mes_numero in padrao['meses_pagamento']:
                div_por_acao = padrao['valores_medios'].get(mes_numero, 0)
                div_projetado = div_por_acao * quantidade
                
                dividendo_total += div_projetado
                detalhes[ticker] = div_projetado
        
        projecao.append({
            'Mês': mes_nome,
            'Mes_Numero': mes_numero,
            'Data': mes_data,
            'Dividendos': dividendo_total,
            'Detalhes': detalhes
        })
    
    return pd.DataFrame(projecao)


def criar_tabela_detalhada_projecao(df_projecao):
    """Cria tabela com breakdown por ativo"""
    
    if df_projecao.empty:
        return pd.DataFrame()
    
    detalhes = []
    
    for _, row in df_projecao.iterrows():
        detalhes_dict = row['Detalhes']
        
        if detalhes_dict:
            breakdown = ', '.join([f"{ticker}: R$ {valor:.2f}" 
                                  for ticker, valor in detalhes_dict.items()])
            ativos_pagantes = len(detalhes_dict)
        else:
            breakdown = "Nenhum pagamento esperado"
            ativos_pagantes = 0
        
        detalhes.append({
            'Mês': row['Mês'],
            'Total': row['Dividendos'],
            'Ativos': ativos_pagantes,
            'Detalhamento': breakdown
        })
    
    return pd.DataFrame(detalhes)


def criar_heatmap_projecao(df_projecao, quantidades):
    """Cria heatmap de pagamentos por ativo/mês"""
    
    if df_projecao.empty or not quantidades:
        return None
    
    meses = df_projecao['Mês'].tolist()
    ativos = sorted(quantidades.keys())
    
    # Criar matriz
    matriz = []
    
    for ativo in ativos:
        linha = []
        for _, row in df_projecao.iterrows():
            detalhes = row['Detalhes']
            valor = detalhes.get(ativo, 0)
            linha.append(valor)
        matriz.append(linha)
    
    fig = go.Figure(data=go.Heatmap(
        z=matriz,
        x=meses,
        y=ativos,
        colorscale='Greens',
        hovertemplate='%{y}<br>%{x}<br>R$ %{z:.2f}<extra></extra>',
        colorbar=dict(title="R$")
    ))
    
    fig.update_layout(
        title='Calendário de Pagamentos Projetados por Ativo',
        xaxis_title='Mês',
        yaxis_title='Ativo',
        height=max(300, len(ativos) * 40),
        hovermode='closest'
    )
    
    return fig


def criar_grafico_projecao_dividendos(df_projecao):
    """Gráfico de barras da projeção mensal"""
    
    if df_projecao.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_projecao['Mês'],
        y=df_projecao['Dividendos'],
        marker_color='#2ecc71',
        hovertemplate='%{x}<br>R$ %{y:.2f}<extra></extra>'
    ))
    
    # Média
    media = df_projecao['Dividendos'].mean()
    fig.add_hline(
        y=media,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Média: R$ {media:.2f}/mês',
        annotation_position='right'
    )
    
    fig.update_layout(
        title='Projeção de Dividendos Mensais (12 meses)',
        xaxis_title='Mês',
        yaxis_title='Dividendos Projetados (R$)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def criar_grafico_pizza_alocacao(df_alocacao):
    """Gráfico de pizza da alocação"""
    
    fig = go.Figure(data=[go.Pie(
        labels=df_alocacao['Ativo'],
        values=df_alocacao['Valor Real'],
        hole=0.3,
        textinfo='label+percent',
        hovertemplate='%{label}<br>R$ %{value:,.2f}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Distribuição do Investimento',
        height=500,
        showlegend=True
    )
    
    return fig


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal"""
    
    st.title("📋 Resumo Executivo")
    st.markdown("Recomendação final personalizada com projeção realista de dividendos")
    st.markdown("---")
    
    # Verificar
    if not st.session_state.portfolio_tickers:
        st.warning("⚠️ Nenhum ativo no portfólio")
        st.info("👉 Vá para **Selecionar Ativos** primeiro")
        st.stop()
    
    if len(st.session_state.portfolio_tickers) < 2:
        st.warning("⚠️ Selecione pelo menos 2 ativos")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.subheader("💰 Investimento")
        
        valor_investimento = st.number_input(
            "Valor a Investir (R$)",
            min_value=1000.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            help="Quanto você pretende investir"
        )
        
        st.markdown("---")
        
        st.subheader("🎯 Objetivo Principal")
        
        objetivo_usuario = st.radio(
            "O que você prioriza?",
            [
                "Deixar o sistema decidir",
                "Máximo retorno ajustado ao risco",
                "Mínima volatilidade",
                "Renda mensal de dividendos"
            ],
            help="Sua escolha determina qual portfólio será recomendado"
        )
        
        st.markdown("---")
        
        st.subheader("⚙️ Avançado")
        
        rf_rate = st.number_input(
            "Taxa Livre de Risco (anual)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_free_rate,
            step=0.0001,
            format="%.4f",
            help="Taxa CDI ou Selic"
        )
        st.session_state.risk_free_rate = rf_rate
        
        st.markdown("---")
        
        btn_gerar = st.button(
            "📊 Gerar Recomendação",
            type="primary",
            use_container_width=True
        )
    
    # Info
    st.info(f"📊 **{len(st.session_state.portfolio_tickers)} ativos** no portfólio")
    
    with st.expander("📋 Ver lista de ativos"):
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            with cols[idx % 5]:
                st.write(f"• {ticker}")
    
    st.markdown("---")
    
    # Gerar
    if btn_gerar:
        
        # Carregar dados
        tickers = st.session_state.portfolio_tickers
        start_date = st.session_state.period_start
        end_date = st.session_state.period_end
        
        price_data, _ = carregar_dados_cache(tickers, start_date, end_date)
        
        if price_data is None or price_data.empty:
            st.error("❌ Dados não disponíveis")
            st.warning("Carregue dados em **Análise de Dividendos** ou **Portfólios Eficientes** primeiro")
            st.stop()
        
        # Limpar
        price_data = price_data.dropna(axis=1, thresh=len(price_data) * 0.8)
        
        if price_data.empty or len(price_data.columns) < 2:
            st.error("❌ Dados insuficientes")
            st.stop()
        
        st.success("✓ Dados carregados do cache")
        
        # Calcular
        returns = calcular_retornos(price_data)
        
        # Perfil
        perfil = analisar_perfil_investidor(valor_investimento)
        
        # Otimizar
        with st.spinner("🧮 Otimizando portfólios..."):
            
            portfolios = {}
            
            # Sharpe
            p_sharpe = otimizar_sharpe_maximo(returns, rf_rate)
            if p_sharpe:
                portfolios['sharpe_maximo'] = p_sharpe
            
            # MinVol
            p_minvol = otimizar_minima_volatilidade(returns)
            if p_minvol:
                portfolios['minima_volatilidade'] = p_minvol
            
            # Dividendos
            metricas_div = st.session_state.get('metricas_dividendos', None)
            p_div = otimizar_dividendos(returns, metricas_div)
            if p_div:
                portfolios['dividendos_regulares'] = p_div
        
        if not portfolios:
            st.error("❌ Falha na otimização")
            st.stop()
        
        st.success(f"✓ {len(portfolios)} portfólios otimizados")
        
        # GERAR RECOMENDAÇÃO
        recomendacao = gerar_recomendacao_por_objetivo(
            objetivo_usuario,
            portfolios,
            perfil
        )
        
        st.markdown("---")
        
        # EXIBIR
        st.success("✅ Análise concluída!")
        st.markdown("---")
        
        # Header
        st.header("🎯 Sua Recomendação Personalizada")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"## {recomendacao['info']['emoji']} {recomendacao['info']['titulo']}")
            st.markdown(f"**{recomendacao['info']['descricao']}**")
            st.markdown(f"*{recomendacao['info']['indicado']}*")
            
            if objetivo_usuario != "Deixar o sistema decidir":
                st.info(f"📌 Baseado no seu objetivo: **{objetivo_usuario}**")
            else:
                st.info(f"📌 Sistema escolheu baseado no perfil: **{perfil.title()}**")
        
        with col2:
            st.metric("Perfil", perfil.title())
            st.metric("Estratégia", recomendacao['portfolio'].replace('_', ' ').title())
        
        st.markdown("---")
        
        # Métricas
        st.subheader("📊 Métricas Esperadas")
        
        portfolio = recomendacao['dados']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Retorno Anual", f"{portfolio['return']:.2%}")
        
        with col2:
            st.metric("Volatilidade", f"{portfolio['volatility']:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{portfolio['sharpe']:.3f}")
        
        with col4:
            ganho = valor_investimento * portfolio['return']
            st.metric("Ganho Esperado (1 ano)", f"R$ {ganho:,.2f}")
        
        st.markdown("---")
        
        # Vantagens
        st.subheader("✨ Por que esta recomendação?")
        
        for vantagem in recomendacao['info']['vantagens']:
            st.markdown(f"✅ {vantagem}")
        
        st.markdown("---")
        
        # Alocação
        st.subheader("💼 Plano de Investimento")
        
        # Preços
        precos_atuais = {}
        for ticker in portfolio['weights'].keys():
            if ticker in price_data.columns:
                precos_atuais[ticker] = float(price_data[ticker].iloc[-1])
        
        # Quantidades
        df_alocacao = calcular_quantidades(
            portfolio['weights'],
            valor_investimento,
            precos_atuais
        )
        
        if not df_alocacao.empty:
            
            # Resumo
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Valor Total", f"R$ {df_alocacao['Valor Real'].sum():,.2f}")
            
            with col2:
                diferenca = valor_investimento - df_alocacao['Valor Real'].sum()
                st.metric("Sobra", f"R$ {diferenca:,.2f}")
            
            with col3:
                st.metric("Ativos", len(df_alocacao))
            
            # Tabela
            st.dataframe(
                df_alocacao.style.format({
                    'Peso': '{:.2%}',
                    'Valor Alvo': 'R$ {:.2f}',
                    'Preço': 'R$ {:.2f}',
                    'Quantidade': '{:.0f}',
                    'Valor Real': 'R$ {:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Gráfico
            fig = criar_grafico_pizza_alocacao(df_alocacao)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # PROJEÇÃO DE DIVIDENDOS
            st.subheader("💵 Projeção de Dividendos Futuros")
            
            st.info("📊 Projeção baseada no **padrão histórico real** de cada ativo (últimos 3 anos)")
            
            with st.spinner("Analisando histórico e projetando..."):
                df_projecao = projetar_dividendos_futuros(
                    list(portfolio['weights'].keys()),
                    portfolio['weights'],
                    valor_investimento,
                    precos_atuais,
                    meses=12
                )
            
            if not df_projecao.empty:
                
                # Métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    media = df_projecao['Dividendos'].mean()
                    st.metric("Média Mensal", f"R$ {media:.2f}")
                
                with col2:
                    total = df_projecao['Dividendos'].sum()
                    st.metric("Total Anual", f"R$ {total:.2f}")
                
                with col3:
                    dy = (total / valor_investimento) * 100
                    st.metric("DY Projetado", f"{dy:.2f}%")
                
                with col4:
                    desvio = df_projecao['Dividendos'].std()
                    st.metric("Desvio Padrão", f"R$ {desvio:.2f}")
                
                # Gráfico mensal
                fig = criar_grafico_projecao_dividendos(df_projecao)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Heatmap
                st.subheader("📅 Calendário: Quando Cada Ativo Paga")
                
                quantidades_map = {}
                for ticker, peso in portfolio['weights'].items():
                    if peso >= 0.01:
                        valor = valor_investimento * peso
                        preco = precos_atuais.get(ticker, 0)
                        if preco > 0:
                            quantidades_map[ticker] = int(valor / preco)
                
                fig_heat = criar_heatmap_projecao(df_projecao, quantidades_map)
                if fig_heat:
                    st.plotly_chart(fig_heat, use_container_width=True)
                
                # Tabela detalhada
                with st.expander("📅 Ver projeção mês a mês com detalhamento"):
                    df_detalhes = criar_tabela_detalhada_projecao(df_projecao)
                    
                    st.dataframe(
                        df_detalhes.style.format({'Total': 'R$ {:.2f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Observações
                st.info("""
                **📌 Como interpretar a projeção:**
                
                ✅ **Meses com barras altas**: Vários ativos pagando simultaneamente  
                ✅ **Meses com barras baixas ou zero**: Poucos ou nenhum pagamento  
                ✅ **Heatmap**: Mostra exatamente quais ativos pagam em cada mês  
                
                ⚠️ **Importante:**
                - Projeção baseada no histórico de 3 anos
                - Cada ativo mantém seu padrão de pagamento
                - Ex: Se COGN3 paga apenas em maio, aparece só em maio
                - Valores podem variar conforme lucros das empresas
                - Use como **estimativa**, não como garantia
                """)
            
            else:
                st.warning("⚠️ Histórico insuficiente para projeção")
            
            st.markdown("---")
            
            # Próximos passos
            st.subheader("📝 Próximos Passos")
            
            st.markdown("""
            **1. Revise a alocação sugerida**
            - Confira ativos e quantidades
            - Verifique se está confortável
            
            **2. Execute as ordens na corretora**
            - Use quantidades exatas da tabela
            - Considere ordens limitadas
            
            **3. Configure recebimento de dividendos**
            - Reinvestimento automático, ou
            - Transferência para conta corrente
            
            **4. Acompanhamento**
            - Monitore mensalmente
            - Compare dividendos reais vs projetados
            - Rebalanceie trimestralmente se necessário
            """)
            
            st.markdown("---")
            
            # Export
            st.subheader("📥 Exportar Documentação")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_alocacao.to_csv(index=False)
                st.download_button(
                    "📊 Alocação (CSV)",
                    data=csv,
                    file_name=f"alocacao_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Resumo completo
                resumo = f"""
RESUMO EXECUTIVO - PORTFOLIO B3
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
{'='*60}

RECOMENDAÇÃO: {recomendacao['info']['titulo']}
Objetivo Selecionado: {objetivo_usuario}
Perfil Identificado: {perfil.title()}
Valor a Investir: R$ {valor_investimento:,.2f}

MÉTRICAS ESPERADAS:
- Retorno Anual: {portfolio['return']:.2%}
- Volatilidade Anual: {portfolio['volatility']:.2%}
- Sharpe Ratio: {portfolio['sharpe']:.3f}
- Ganho Esperado (1 ano): R$ {ganho:,.2f}

PROJEÇÃO DE DIVIDENDOS (12 meses):
- Média Mensal: R$ {media:.2f}
- Total Anual: R$ {total:.2f}
- DY Projetado: {dy:.2f}%
- Desvio Padrão: R$ {desvio:.2f}

ALOCAÇÃO DETALHADA:
{df_alocacao.to_string(index=False)}

Valor Total Alocado: R$ {df_alocacao['Valor Real'].sum():,.2f}
Sobra (não investida): R$ {diferenca:,.2f}

PROJEÇÃO MENSAL DE DIVIDENDOS:
{df_detalhes.to_string(index=False)}

OBSERVAÇÕES IMPORTANTES:
- Projeção baseada em padrão histórico de 3 anos
- Cada ativo mantém seu calendário específico de pagamentos
- Valores são estimativas baseadas em médias históricas
- Rentabilidade passada não garante resultados futuros
- Esta análise não constitui recomendação de investimento
- Consulte um profissional certificado antes de investir

Gerado por: Portfolio B3 App
                """
                
                st.download_button(
                    "📄 Resumo Completo (TXT)",
                    data=resumo,
                    file_name=f"resumo_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        else:
            st.warning("⚠️ Erro ao calcular alocação")
        
        st.markdown("---")
        
        # Disclaimer
        st.warning("""
        **⚠️ Aviso Legal**
        
        Esta recomendação é baseada em análise quantitativa de dados históricos e não 
        constitui consultoria financeira ou recomendação de investimento. Rentabilidade 
        passada não garante resultados futuros. Consulte um profissional certificado 
        (AAI, CFP, CGA) antes de tomar decisões de investimento. Investimentos em renda 
        variável envolvem riscos, incluindo perda de capital.
        """)
    
    else:
        st.info("👈 Configure seu objetivo e valor, depois clique em **Gerar Recomendação**")
        
        # Info
        with st.expander("ℹ️ Como funciona a recomendação"):
            st.markdown("""
            ### 🎯 Sistema de Recomendação
            
            **Análise de Perfil:**
            - Conservador: < R$ 10.000
            - Moderado: R$ 10.000 - R$ 50.000
            - Agressivo: > R$ 50.000
            
            **Objetivos:**
            
            1. **Deixar o sistema decidir**: Análise automática baseada no perfil
            2. **Máximo Sharpe**: Melhor relação risco-retorno
            3. **Mínima Volatilidade**: Menor risco possível
            4. **Dividendos**: Foco em renda mensal regular
            
            **Projeção de Dividendos:**
            - Analisa histórico de 3 anos de cada ativo
            - Identifica em QUAIS MESES cada ativo costuma pagar
            - Usa valor MÉDIO histórico de cada mês
            - Projeta 12 meses respeitando o calendário real
            
            **Exemplo:**
            - Se COGN3 historicamente paga apenas em maio
            - Projeção mostrará dividendos APENAS em maio
            - Valor será a média dos pagamentos de maio dos últimos anos
            """)


if __name__ == "__main__":
    main()
