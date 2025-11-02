"""
📋 Resumo Executivo
Recomendação final personalizada com plano de investimento e projeção de dividendos
"""

import streamlit as st
import pandas as pd
import numpy as np
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


def otimizar_portfolio(returns, objetivo='sharpe', rf_rate=0.1175):
    """
    Otimiza portfólio baseado em objetivo
    
    Args:
        returns: DataFrame de retornos
        objetivo: 'sharpe', 'minvol', 'dividendos'
        rf_rate: Taxa livre de risco
        
    Returns:
        Dict com pesos e métricas
    """
    n_assets = len(returns.columns)
    
    def portfolio_return(weights):
        return np.sum(returns.mean() * weights) * 252
    
    def portfolio_vol(weights):
        cov = returns.cov() * 252
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    
    def sharpe_ratio(weights):
        ret = portfolio_return(weights)
        vol = portfolio_vol(weights)
        return (ret - rf_rate) / vol if vol > 0 else 0
    
    # Definir objetivo
    if objetivo == 'sharpe':
        objective = lambda w: -sharpe_ratio(w)
    elif objetivo == 'minvol':
        objective = lambda w: portfolio_vol(w)
    else:  # dividendos
        objective = lambda w: -portfolio_return(w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objective,
        initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        return None
    
    weights = result.x
    
    return {
        'weights': dict(zip(returns.columns, weights)),
        'return': portfolio_return(weights),
        'volatility': portfolio_vol(weights),
        'sharpe': sharpe_ratio(weights)
    }


# ==========================================
# PROJEÇÃO DE DIVIDENDOS
# ==========================================

def projetar_dividendos_futuros(tickers, weights, valor_investimento, precos_atuais, meses=12):
    """
    Projeta dividendos mensais futuros baseado no histórico
    
    Args:
        tickers: Lista de tickers
        weights: Dict com pesos
        valor_investimento: Valor total investido
        precos_atuais: Dict com preços atuais
        meses: Número de meses para projetar
        
    Returns:
        DataFrame com projeção mensal
    """
    # Buscar dividendos históricos (últimos 2 anos)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    dividendos_historicos = {}
    
    for ticker in tickers:
        try:
            divs = data.get_dividends(ticker, start_date, end_date)
            if not divs.empty:
                dividendos_historicos[ticker] = divs
        except:
            continue
    
    if not dividendos_historicos:
        return pd.DataFrame()
    
    # Calcular média mensal por ativo
    medias_mensais = {}
    
    for ticker, divs_df in dividendos_historicos.items():
        # Agrupar por mês
        divs_df['mes'] = pd.to_datetime(divs_df['data']).dt.to_period('M')
        divs_mensais = divs_df.groupby('mes')['valor'].sum()
        
        # Média mensal
        if len(divs_mensais) > 0:
            media_mensal = divs_mensais.mean()
            medias_mensais[ticker] = media_mensal
        else:
            medias_mensais[ticker] = 0
    
    # Calcular quantidades de cada ativo
    quantidades = {}
    
    for ticker, peso in weights.items():
        if peso < 0.01:
            continue
        
        valor_alocar = valor_investimento * peso
        preco = precos_atuais.get(ticker, 0)
        
        if preco > 0:
            quantidade = int(valor_alocar / preco)
            quantidades[ticker] = quantidade
    
    # Projetar dividendos mensais
    projecao = []
    data_inicio_projecao = datetime.now()
    
    for i in range(meses):
        mes_data = data_inicio_projecao + timedelta(days=30 * i)
        mes_nome = mes_data.strftime('%Y-%m')
        
        dividendo_total_mes = 0
        detalhes_mes = {}
        
        for ticker, quantidade in quantidades.items():
            div_mensal_medio = medias_mensais.get(ticker, 0)
            div_projetado = div_mensal_medio * quantidade
            
            dividendo_total_mes += div_projetado
            
            if div_projetado > 0:
                detalhes_mes[ticker] = div_projetado
        
        projecao.append({
            'Mês': mes_nome,
            'Data': mes_data,
            'Dividendos': dividendo_total_mes,
            'Detalhes': detalhes_mes
        })
    
    df_projecao = pd.DataFrame(projecao)
    
    return df_projecao


def criar_grafico_projecao_dividendos(df_projecao):
    """Cria gráfico de projeção de dividendos mensais"""
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_projecao['Mês'],
        y=df_projecao['Dividendos'],
        marker_color='#2ecc71',
        hovertemplate='%{x}<br>R$ %{y:.2f}<extra></extra>'
    ))
    
    # Linha de média
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


# ==========================================
# RECOMENDAÇÃO BASEADA EM OBJETIVO
# ==========================================

def gerar_recomendacao_por_objetivo(objetivo_usuario, portfolios, perfil, metricas_dividendos):
    """
    Gera recomendação baseada no objetivo escolhido pelo usuário
    
    Args:
        objetivo_usuario: String com objetivo
        portfolios: Dict com portfólios otimizados
        perfil: Perfil do investidor
        metricas_dividendos: DataFrame com métricas
        
    Returns:
        Dict com recomendação
    """
    # Mapear objetivo para portfólio
    if objetivo_usuario == "Máximo retorno ajustado ao risco":
        portfolio_escolhido = 'sharpe_maximo'
        
    elif objetivo_usuario == "Mínima volatilidade":
        portfolio_escolhido = 'minima_volatilidade'
        
    elif objetivo_usuario == "Renda mensal de dividendos":
        portfolio_escolhido = 'dividendos_regulares' if 'dividendos_regulares' in portfolios else 'sharpe_maximo'
        
    else:  # "Deixar o sistema decidir"
        # Decidir baseado no perfil e métricas
        if perfil == 'conservador':
            portfolio_escolhido = 'minima_volatilidade'
        elif perfil == 'agressivo':
            portfolio_escolhido = 'sharpe_maximo'
        else:  # moderado
            # Escolher o com melhor Sharpe
            melhor_sharpe = max(portfolios.items(), key=lambda x: x[1]['sharpe'])
            portfolio_escolhido = melhor_sharpe[0]
    
    # Verificar se portfólio existe
    if portfolio_escolhido not in portfolios:
        portfolio_escolhido = list(portfolios.keys())[0]
    
    # Informações explicativas
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
            ]
        },
        'minima_volatilidade': {
            'titulo': '🛡️ Portfólio de Mínima Volatilidade',
            'descricao': 'Máxima estabilidade e menor risco possível',
            'indicado': 'Investidores conservadores que priorizam preservação de capital',
            'vantagens': [
                'Menor oscilação de preços',
                'Ideal para perfil conservador',
                'Proteção em momentos de crise',
                'Maior previsibilidade'
            ]
        },
        'dividendos_regulares': {
            'titulo': '💰 Portfólio de Dividendos Regulares',
            'descricao': 'Foco em renda passiva mensal consistente',
            'indicado': 'Investidores que buscam fluxo de caixa regular',
            'vantagens': [
                'Renda mensal previsível',
                'Bons pagadores de dividendos',
                'Estratégia de longo prazo',
                'Ideal para aposentadoria'
            ]
        }
    }
    
    return {
        'portfolio': portfolio_escolhido,
        'dados': portfolios[portfolio_escolhido],
        'info': explicacoes[portfolio_escolhido],
        'motivo_escolha': objetivo_usuario
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
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal"""
    
    st.title("📋 Resumo Executivo")
    st.markdown("Recomendação final personalizada com plano de investimento detalhado")
    st.markdown("---")
    
    # Verificar portfólio
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
            step=1000.0
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
            help="Selecione seu objetivo de investimento"
        )
        
        st.markdown("---")
        
        rf_rate = st.number_input(
            "Taxa Livre de Risco",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_free_rate,
            step=0.0001,
            format="%.4f"
        )
        
        st.markdown("---")
        
        btn_gerar = st.button(
            "📊 Gerar Recomendação",
            type="primary",
            use_container_width=True
        )
    
    # Info
    st.info(f"📊 **{len(st.session_state.portfolio_tickers)} ativos** no portfólio")
    
    with st.expander("📋 Ver lista"):
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            with cols[idx % 5]:
                st.write(f"• {ticker}")
    
    st.markdown("---")
    
    # Gerar recomendação
    if btn_gerar:
        
        # Carregar dados
        tickers = st.session_state.portfolio_tickers
        start_date = st.session_state.period_start
        end_date = st.session_state.period_end
        
        price_data, _ = carregar_dados_cache(tickers, start_date, end_date)
        
        if price_data is None or price_data.empty:
            st.error("❌ Dados não disponíveis. Carregue dados em outra página primeiro.")
            st.stop()
        
        # Limpar
        price_data = price_data.dropna(axis=1, thresh=len(price_data) * 0.8)
        
        if price_data.empty or len(price_data.columns) < 2:
            st.error("❌ Dados insuficientes")
            st.stop()
        
        st.success("✓ Dados carregados do cache")
        
        # Calcular retornos
        returns = calcular_retornos(price_data)
        
        # Otimizar portfólios
        with st.spinner("🧮 Otimizando portfólios..."):
            
            portfolios = {}
            
            # Sharpe Máximo
            p_sharpe = otimizar_portfolio(returns, 'sharpe', rf_rate)
            if p_sharpe:
                portfolios['sharpe_maximo'] = p_sharpe
            
            # Mínima Volatilidade
            p_minvol = otimizar_portfolio(returns, 'minvol', rf_rate)
            if p_minvol:
                portfolios['minima_volatilidade'] = p_minvol
            
            # Dividendos (usar sharpe mas com foco em DY)
            p_div = otimizar_portfolio(returns, 'dividendos', rf_rate)
            if p_div:
                portfolios['dividendos_regulares'] = p_div
        
        if not portfolios:
            st.error("❌ Falha na otimização")
            st.stop()
        
        st.success(f"✓ {len(portfolios)} portfólios otimizados")
        
        # Determinar perfil
        perfil = analisar_perfil_investidor(valor_investimento)
        
        # Carregar métricas de dividendos
        metricas_dividendos = st.session_state.get('metricas_dividendos', None)
        
        # GERAR RECOMENDAÇÃO BASEADA NO OBJETIVO
        recomendacao = gerar_recomendacao_por_objetivo(
            objetivo_usuario,
            portfolios,
            perfil,
            metricas_dividendos
        )
        
        if not recomendacao:
            st.error("❌ Não foi possível gerar recomendação")
            st.stop()
        
        # ==========================================
        # EXIBIR RECOMENDAÇÃO
        # ==========================================
        
        st.success("✅ Análise concluída!")
        st.markdown("---")
        
        # Header
        st.header("🎯 Sua Recomendação Personalizada")
        
        # Explicar escolha
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"## {recomendacao['info']['titulo']}")
            st.markdown(f"**{recomendacao['info']['descricao']}**")
            st.markdown(f"*{recomendacao['info']['indicado']}*")
            
            if objetivo_usuario != "Deixar o sistema decidir":
                st.info(f"📌 Recomendação baseada no seu objetivo: **{objetivo_usuario}**")
            else:
                st.info(f"📌 Sistema recomendou baseado no seu perfil: **{perfil.title()}**")
        
        with col2:
            st.metric("Perfil", perfil.title())
            st.metric("Portfólio", recomendacao['portfolio'].replace('_', ' ').title())
        
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
            ganho_esperado = valor_investimento * portfolio['return']
            st.metric("Ganho Esperado (1 ano)", f"R$ {ganho_esperado:,.2f}")
        
        st.markdown("---")
        
        # Vantagens
        st.subheader("✨ Por que esta recomendação?")
        
        for vantagem in recomendacao['info']['vantagens']:
            st.markdown(f"✅ {vantagem}")
        
        st.markdown("---")
        
        # Alocação
        st.subheader("💼 Plano de Investimento")
        
        # Obter preços atuais
        precos_atuais = {}
        for ticker in portfolio['weights'].keys():
            if ticker in price_data.columns:
                precos_atuais[ticker] = float(price_data[ticker].iloc[-1])
        
        # Calcular quantidades
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
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Pie(
                labels=df_alocacao['Ativo'],
                values=df_alocacao['Valor Real'],
                hole=0.3,
                textinfo='label+percent',
                hovertemplate='%{label}<br>R$ %{value:,.2f}<extra></extra>'
            )])
            
            fig.update_layout(title='Distribuição do Investimento', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # PROJEÇÃO DE DIVIDENDOS
            st.subheader("💵 Projeção de Dividendos Futuros")
            
            with st.spinner("Calculando projeção de dividendos..."):
                df_projecao = projetar_dividendos_futuros(
                    list(portfolio['weights'].keys()),
                    portfolio['weights'],
                    valor_investimento,
                    precos_atuais,
                    meses=12
                )
            
            if not df_projecao.empty:
                
                # Métricas de dividendos
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    media_mensal = df_projecao['Dividendos'].mean()
                    st.metric("Média Mensal", f"R$ {media_mensal:.2f}")
                
                with col2:
                    total_anual = df_projecao['Dividendos'].sum()
                    st.metric("Total Anual Projetado", f"R$ {total_anual:.2f}")
                
                with col3:
                    dy_projetado = (total_anual / valor_investimento) * 100
                    st.metric("DY Projetado", f"{dy_projetado:.2f}%")
                
                with col4:
                    desvio = df_projecao['Dividendos'].std()
                    st.metric("Desvio Padrão", f"R$ {desvio:.2f}")
                
                # Gráfico de projeção
                fig = criar_grafico_projecao_dividendos(df_projecao)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela mensal detalhada
                with st.expander("📅 Ver projeção mês a mês"):
                    st.dataframe(
                        df_projecao[['Mês', 'Dividendos']].style.format({
                            'Dividendos': 'R$ {:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            
            else:
                st.warning("⚠️ Não foi possível projetar dividendos (dados históricos insuficientes)")
            
            st.markdown("---")
            
            # Próximos passos
            st.subheader("📝 Próximos Passos")
            
            st.markdown("""
            **1. Revise a alocação**
            - Confira os ativos e quantidades sugeridas
            - Verifique se está confortável com a distribuição
            
            **2. Execute as ordens**
            - Acesse sua corretora
            - Use as quantidades exatas da tabela
            - Considere ordens limitadas para melhores preços
            
            **3. Acompanhamento**
            - Monitore mensalmente
            - Rebalanceie trimestralmente se necessário
            - Mantenha disciplina na estratégia
            
            **4. Dividendos**
            - Configure reinvestimento automático, ou
            - Use a renda para seus objetivos
            - Acompanhe os pagamentos mensais
            """)
            
            st.markdown("---")
            
            # Export
            st.subheader("📥 Exportar Recomendação")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_alocacao.to_csv(index=False)
                st.download_button(
                    "📊 Baixar Alocação (CSV)",
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
=====================================

RECOMENDAÇÃO: {recomendacao['info']['titulo']}
Objetivo: {objetivo_usuario}
Perfil: {perfil.title()}
Valor Investido: R$ {valor_investimento:,.2f}

MÉTRICAS ESPERADAS:
- Retorno Anual: {portfolio['return']:.2%}
- Volatilidade: {portfolio['volatility']:.2%}
- Sharpe Ratio: {portfolio['sharpe']:.3f}
- Ganho Esperado (1 ano): R$ {ganho_esperado:,.2f}

PROJEÇÃO DE DIVIDENDOS:
- Média Mensal: R$ {media_mensal:.2f}
- Total Anual: R$ {total_anual:.2f}
- DY Projetado: {dy_projetado:.2f}%

ALOCAÇÃO DETALHADA:
{df_alocacao.to_string(index=False)}

Total Alocado: R$ {df_alocacao['Valor Real'].sum():,.2f}
Sobra: R$ {diferenca:,.2f}

PROJEÇÃO MENSAL:
{df_projecao[['Mês', 'Dividendos']].to_string(index=False)}

OBSERVAÇÕES:
- Esta recomendação é baseada em análise quantitativa
- Não constitui consultoria financeira
- Consulte um profissional certificado
- Rentabilidade passada não garante resultados futuros
                """
                
                st.download_button(
                    "📄 Baixar Resumo Completo (TXT)",
                    data=resumo,
                    file_name=f"resumo_executivo_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        else:
            st.warning("⚠️ Não foi possível calcular alocação")
        
        st.markdown("---")
        
        # Disclaimer
        st.warning("""
        **⚠️ Aviso Legal**
        
        Esta recomendação é baseada em análise quantitativa histórica e não constitui 
        consultoria financeira. Rentabilidade passada não garante resultados futuros. 
        Consulte um profissional certificado antes de tomar decisões de investimento. 
        Investimentos em renda variável envolvem riscos de perda de capital.
        """)
    
    else:
        st.info("👈 Configure seu objetivo e valor na barra lateral, depois clique em **Gerar Recomendação**")
        
        # Informações
        with st.expander("ℹ️ Como funciona"):
            st.markdown("""
            ### 🎯 Sistema de Recomendação Inteligente
            
            **1. Análise de Perfil**
            - Baseado no valor a investir
            - Conservador: < R$ 10.000
            - Moderado: R$ 10.000 - R$ 50.000
            - Agressivo: > R$ 50.000
            
            **2. Objetivos Disponíveis**
            
            **Deixar o sistema decidir:**
            - Sistema analisa seu perfil
            - Recomenda automaticamente
            - Combina análise técnica e fundamental
            
            **Máximo retorno ajustado ao risco:**
            - Portfólio de Máximo Sharpe
            - Melhor relação risco/retorno
            - Ideal para crescimento
            
            **Mínima volatilidade:**
            - Portfólio mais estável
            - Menor risco possível
            - Ideal para preservação
            
            **Renda mensal de dividendos:**
            - Foco em dividend yield
            - Projeção de renda mensal
            - Ideal para renda passiva
            
            **3. Plano de Investimento**
            - Quantidades exatas a comprar
            - Valores por ativo
            - Preços atuais de referência
            
            **4. Projeção de Dividendos**
            - Baseada em histórico de 2 anos
            - Média mensal por ativo
            - Projeção para 12 meses
            - DY esperado
            """)


if __name__ == "__main__":
    main()
