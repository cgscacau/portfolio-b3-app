"""
📋 Resumo Executivo
Recomendação final personalizada com plano de investimento detalhado
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import sys
from pathlib import Path

# Configurar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core.init import init_all
from core.cache import carregar_dados_cache

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


def otimizar_portfolio(returns, objetivo='sharpe', rf_rate=0.1175, target_risk=None):
    """
    Otimiza portfólio baseado em objetivo
    
    Args:
        returns: DataFrame de retornos
        objetivo: 'sharpe', 'minvol', 'target_return'
        rf_rate: Taxa livre de risco
        target_risk: Risco alvo (para target_return)
        
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
        return (ret - rf_rate) / vol
    
    # Configurar otimização baseada no objetivo
    if objetivo == 'sharpe':
        objective = lambda w: -sharpe_ratio(w)
    elif objetivo == 'minvol':
        objective = lambda w: portfolio_vol(w)
    else:
        objective = lambda w: -portfolio_return(w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    if target_risk and objetivo == 'target_return':
        constraints.append({'type': 'ineq', 'fun': lambda w: target_risk - portfolio_vol(w)})
    
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
# ANÁLISE E RECOMENDAÇÃO
# ==========================================

def analisar_perfil_investidor(valor_investimento):
    """
    Determina perfil do investidor baseado no valor
    
    Args:
        valor_investimento: Valor a investir
        
    Returns:
        String com perfil
    """
    if valor_investimento < 10000:
        return 'conservador'
    elif valor_investimento < 50000:
        return 'moderado'
    else:
        return 'agressivo'


def recomendar_portfolio(portfolios, perfil, metricas_dividendos=None):
    """
    Recomenda o melhor portfólio baseado em análise técnica e fundamental
    
    Args:
        portfolios: Dict com portfólios otimizados
        perfil: Perfil do investidor
        metricas_dividendos: DataFrame com métricas de dividendos (opcional)
        
    Returns:
        Dict com recomendação
    """
    # Scores para cada portfólio
    scores = {}
    
    # Máximo Sharpe - Melhor relação risco/retorno
    if 'sharpe_maximo' in portfolios:
        p = portfolios['sharpe_maximo']
        score = 0
        
        # Análise técnica
        score += p['sharpe'] * 30  # Sharpe é muito importante
        score += p['return'] * 20   # Retorno esperado
        score -= p['volatility'] * 10  # Penalizar volatilidade
        
        # Ajuste por perfil
        if perfil == 'agressivo':
            score += 15  # Agressivos preferem retorno
        elif perfil == 'moderado':
            score += 25  # Moderados adoram Sharpe
        
        scores['sharpe_maximo'] = score
    
    # Mínima Volatilidade - Mais estável
    if 'minima_volatilidade' in portfolios:
        p = portfolios['minima_volatilidade']
        score = 0
        
        # Análise técnica
        score -= p['volatility'] * 40  # Volatilidade baixa é ótimo
        score += p['return'] * 15
        score += p['sharpe'] * 20
        
        # Ajuste por perfil
        if perfil == 'conservador':
            score += 30  # Conservadores preferem estabilidade
        elif perfil == 'moderado':
            score += 15
        
        scores['minima_volatilidade'] = score
    
    # Dividendos Regulares - Renda passiva
    if 'dividendos_regulares' in portfolios and metricas_dividendos is not None:
        p = portfolios['dividendos_regulares']
        score = 0
        
        # Análise fundamental (dividendos)
        dy_medio = metricas_dividendos['dy_anual'].mean()
        regularidade_media = metricas_dividendos['regularidade'].mean()
        
        score += dy_medio * 2  # DY alto é bom
        score += regularidade_media * 0.3  # Regularidade é importante
        score += p['return'] * 10
        
        # Ajuste por perfil
        if perfil == 'conservador':
            score += 20  # Conservadores gostam de renda
        elif perfil == 'moderado':
            score += 10
        
        scores['dividendos_regulares'] = score
    
    # Selecionar melhor
    if not scores:
        return None
    
    melhor = max(scores.items(), key=lambda x: x[1])
    
    # Explicação da escolha
    explicacoes = {
        'sharpe_maximo': {
            'titulo': '🎯 Portfólio de Máximo Sharpe Ratio',
            'descricao': 'Melhor relação risco-retorno disponível',
            'indicado': 'Investidores que buscam eficiência e crescimento balanceado',
            'vantagens': [
                'Otimiza retorno ajustado ao risco',
                'Equilíbrio entre ganhos e volatilidade',
                'Estratégia comprovada pela teoria moderna'
            ]
        },
        'minima_volatilidade': {
            'titulo': '🛡️ Portfólio de Mínima Volatilidade',
            'descricao': 'Máxima estabilidade e menor risco possível',
            'indicado': 'Investidores conservadores que priorizam preservação de capital',
            'vantagens': [
                'Menor oscilação de preços',
                'Ideal para perfil conservador',
                'Proteção em momentos de crise'
            ]
        },
        'dividendos_regulares': {
            'titulo': '💰 Portfólio de Dividendos Regulares',
            'descricao': 'Foco em renda passiva mensal consistente',
            'indicado': 'Investidores que buscam fluxo de caixa regular',
            'vantagens': [
                'Renda mensal previsível',
                'Bons pagadores de dividendos',
                'Estratégia de longo prazo'
            ]
        }
    }
    
    return {
        'portfolio': melhor[0],
        'score': melhor[1],
        'dados': portfolios[melhor[0]],
        'info': explicacoes[melhor[0]]
    }


def calcular_quantidades(weights, valor_investimento, precos_atuais):
    """
    Calcula quantidades a comprar de cada ativo
    
    Args:
        weights: Dict com pesos
        valor_investimento: Valor total
        precos_atuais: Dict com preços
        
    Returns:
        DataFrame com quantidades
    """
    alocacoes = []
    
    for ticker, peso in weights.items():
        if peso < 0.01:  # Ignorar < 1%
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
    
    # Verificar se há portfólio
    if not st.session_state.portfolio_tickers:
        st.warning("⚠️ Nenhum ativo no portfólio")
        st.info("👉 Vá para **Selecionar Ativos** primeiro")
        st.stop()
    
    if len(st.session_state.portfolio_tickers) < 2:
        st.warning("⚠️ Selecione pelo menos 2 ativos para otimização")
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
        
        st.subheader("📊 Objetivo")
        
        objetivo_usuario = st.radio(
            "Prioridade",
            [
                "Deixar o sistema decidir",
                "Máximo retorno ajustado ao risco",
                "Mínima volatilidade",
                "Renda mensal de dividendos"
            ]
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
        
        price_data, dividend_data = carregar_dados_cache(tickers, start_date, end_date)
        
        if price_data is None or price_data.empty:
            st.error("❌ Nenhum portfólio otimizado disponível")
            st.warning("Por favor, otimize pelo menos um portfólio nas páginas anteriores:")
            st.info("• **Portfólios Eficientes** - para Sharpe Máximo")
            st.info("• **Sharpe e MinVol** - para comparação detalhada")
            st.stop()
        
        # Limpar dados
        price_data = price_data.dropna(axis=1, thresh=len(price_data) * 0.8)
        
        if price_data.empty or len(price_data.columns) < 2:
            st.error("❌ Dados insuficientes")
            st.stop()
        
        st.success("✓ Dados carregados")
        
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
        
        if not portfolios:
            st.error("❌ Falha na otimização")
            st.stop()
        
        # Determinar perfil
        perfil = analisar_perfil_investidor(valor_investimento)
        
        # Carregar métricas de dividendos se disponível
        metricas_dividendos = st.session_state.get('metricas_dividendos', None)
        
        # Recomendar
        with st.spinner("🎯 Analisando e gerando recomendação..."):
            recomendacao = recomendar_portfolio(portfolios, perfil, metricas_dividendos)
        
        if not recomendacao:
            st.error("❌ Não foi possível gerar recomendação")
            st.stop()
        
        # ==========================================
        # EXIBIR RECOMENDAÇÃO
        # ==========================================
        
        st.success("✅ Análise concluída!")
        
        st.markdown("---")
        
        # Header da recomendação
        st.header("🎯 Recomendação Final")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"## {recomendacao['info']['titulo']}")
            st.markdown(f"**{recomendacao['info']['descricao']}**")
            st.markdown(f"*{recomendacao['info']['indicado']}*")
        
        with col2:
            st.metric("Perfil Identificado", perfil.title())
            st.metric("Score de Adequação", f"{recomendacao['score']:.1f}")
        
        st.markdown("---")
        
        # Métricas do portfólio recomendado
        st.subheader("📊 Métricas do Portfólio Recomendado")
        
        portfolio = recomendacao['dados']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Retorno Esperado (anual)",
                f"{portfolio['return']:.2%}",
                help="Retorno anualizado esperado"
            )
        
        with col2:
            st.metric(
                "Volatilidade (anual)",
                f"{portfolio['volatility']:.2%}",
                help="Risco anualizado"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{portfolio['sharpe']:.3f}",
                help="Retorno ajustado ao risco"
            )
        
        with col4:
            # Projeção de ganho
            ganho_esperado = valor_investimento * portfolio['return']
            st.metric(
                "Ganho Esperado (1 ano)",
                f"R$ {ganho_esperado:,.2f}",
                help="Projeção baseada no retorno esperado"
            )
        
        st.markdown("---")
        
        # Vantagens
        st.subheader("✨ Por que esta recomendação?")
        
        for vantagem in recomendacao['info']['vantagens']:
            st.markdown(f"✅ {vantagem}")
        
        st.markdown("---")
        
        # Alocação
        st.subheader("💼 Alocação Recomendada")
        
        # Obter preços atuais
        precos_atuais = {}
        for ticker in portfolio['weights'].keys():
            if ticker in price_data.columns:
                precos_atuais[ticker] = price_data[ticker].iloc[-1]
        
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
                st.metric("Valor Total Alocado", f"R$ {df_alocacao['Valor Real'].sum():,.2f}")
            
            with col2:
                diferenca = valor_investimento - df_alocacao['Valor Real'].sum()
                st.metric("Diferença (sobra)", f"R$ {diferenca:,.2f}")
            
            with col3:
                st.metric("Número de Ativos", len(df_alocacao))
            
            st.markdown("---")
            
            # Tabela detalhada
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
            
            st.markdown("---")
            
            # Gráfico de alocação
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Pie(
                labels=df_alocacao['Ativo'],
                values=df_alocacao['Valor Real'],
                hole=0.3,
                textinfo='label+percent',
                hovertemplate='%{label}<br>R$ %{value:,.2f}<br>%{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title='Distribuição do Investimento',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Não foi possível calcular quantidades")
        
        st.markdown("---")
        
        # Próximos passos
        st.subheader("📝 Próximos Passos")
        
        st.markdown("""
        **1. Revise a alocação**
        - Verifique se os ativos e quantidades fazem sentido para você
        - Considere ajustar baseado em suas convicções pessoais
        
        **2. Abra ordens na sua corretora**
        - Use as quantidades exatas da tabela acima
        - Considere fazer ordens limitadas para melhores preços
        
        **3. Acompanhamento**
        - Monitore mensalmente a performance
        - Rebalanceie quando necessário (sugestão: trimestral)
        - Mantenha disciplina na estratégia
        
        **4. Documentação**
        - Exporte esta recomendação (botão abaixo)
        - Guarde para referência futura
        """)
        
        st.markdown("---")
        
        # Export
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV
            csv = df_alocacao.to_csv(index=False)
            st.download_button(
                "📥 Baixar Alocação (CSV)",
                data=csv,
                file_name=f"alocacao_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Texto resumo
            resumo_texto = f"""
RESUMO EXECUTIVO - PORTFOLIO B3
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}

RECOMENDAÇÃO: {recomendacao['info']['titulo']}
Perfil: {perfil.title()}
Valor Investido: R$ {valor_investimento:,.2f}

MÉTRICAS:
- Retorno Esperado: {portfolio['return']:.2%} a.a.
- Volatilidade: {portfolio['volatility']:.2%} a.a.
- Sharpe Ratio: {portfolio['sharpe']:.3f}

ALOCAÇÃO:
{df_alocacao.to_string(index=False)}

Total Alocado: R$ {df_alocacao['Valor Real'].sum():,.2f}
Sobra: R$ {valor_investimento - df_alocacao['Valor Real'].sum():,.2f}
            """
            
            st.download_button(
                "📄 Baixar Resumo (TXT)",
                data=resumo_texto,
                file_name=f"resumo_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Disclaimer
        st.warning("""
        **⚠️ Aviso Legal**
        
        Esta recomendação é baseada em análise quantitativa e não constitui consultoria financeira.
        Rentabilidade passada não garante resultados futuros. Consulte um profissional certificado
        antes de tomar decisões de investimento. Investimentos em renda variável envolvem riscos.
        """)
    
    else:
        st.info("👈 Configure o valor a investir na barra lateral e clique em **Gerar Recomendação**")
        
        # Informações
        with st.expander("ℹ️ Como funciona a recomendação"):
            st.markdown("""
            ### 🎯 Sistema de Recomendação Inteligente
            
            O sistema analisa múltiplos fatores para recomendar o melhor portfólio:
            
            **Análise Técnica:**
            - Retorno esperado
            - Volatilidade (risco)
            - Sharpe Ratio
            - Correlação entre ativos
            
            **Análise Fundamental:**
            - Dividend Yield
            - Regularidade de dividendos
            - Qualidade dos ativos
            
            **Perfil do Investidor:**
            - Conservador: < R$ 10.000
            - Moderado: R$ 10.000 - R$ 50.000
            - Agressivo: > R$ 50.000
            
            **Score de Adequação:**
            - Combina todos os fatores
            - Ajusta por perfil
            - Recomenda o mais adequado
            
            ### 📊 Portfólios Disponíveis
            
            **Máximo Sharpe:** Melhor relação risco-retorno  
            **Mínima Volatilidade:** Menor risco possível  
            **Dividendos Regulares:** Foco em renda passiva (se disponível)
            """)


if __name__ == "__main__":
    main()
