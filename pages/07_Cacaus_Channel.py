"""
Cacau's Channel - Análise Técnica Multi-Timeframe
Detecta convergência entre timeframes diário e semanal
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importar módulos
from core.data import get_price_history
from core.cache import cache_manager
from core.email_alerts import (
    enviar_alerta_oportunidades,
    testar_configuracao_email,
    enviar_email_teste
)


# ==========================================
# FUNÇÕES DE CÁLCULO DO INDICADOR
# ==========================================

def calcular_cacaus_channel(df, periodo_superior=20, periodo_inferior=30, ema_periodo=9):
    """
    Calcula o indicador Cacau's Channel
    
    Args:
        df: DataFrame com preços (colunas: Open, High, Low, Close)
        periodo_superior: Período para linha superior
        periodo_inferior: Período para linha inferior
        ema_periodo: Período da EMA
        
    Returns:
        DataFrame com colunas adicionais do indicador
    """
    df = df.copy()
    
    # Linha Superior - Máxima dos últimos N períodos
    df['linha_superior'] = df['Close'].rolling(window=periodo_superior).max()
    
    # Linha Inferior - Mínima dos últimos N períodos
    df['linha_inferior'] = df['Close'].rolling(window=periodo_inferior).min()
    
    # Linha Média
    df['linha_media'] = (df['linha_superior'] + df['linha_inferior']) / 2
    
    # EMA da Linha Média
    df['ema_media'] = df['linha_media'].ewm(span=ema_periodo, adjust=False).mean()
    
    # Sinal: 1 = Compra (média > ema), -1 = Venda (média < ema), 0 = Neutro
    df['sinal'] = 0
    df.loc[df['linha_media'] > df['ema_media'], 'sinal'] = 1
    df.loc[df['linha_media'] < df['ema_media'], 'sinal'] = -1
    
    return df


def resample_para_semanal(df):
    """
    Converte dados diários para semanais
    
    Args:
        df: DataFrame com dados diários
        
    Returns:
        DataFrame com dados semanais
    """
    df_semanal = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return df_semanal


def detectar_convergencia(df_diario, df_semanal):
    """
    Detecta convergência entre timeframes
    
    Args:
        df_diario: DataFrame com indicador no diário
        df_semanal: DataFrame com indicador no semanal
        
    Returns:
        Dict com resultado da convergência
    """
    # Pegar último sinal de cada timeframe
    sinal_diario = df_diario['sinal'].iloc[-1] if not df_diario.empty else 0
    sinal_semanal = df_semanal['sinal'].iloc[-1] if not df_semanal.empty else 0
    
    # Verificar convergência
    convergente = (sinal_diario == sinal_semanal) and (sinal_diario != 0)
    
    if convergente:
        direcao = 'COMPRA' if sinal_diario == 1 else 'VENDA'
    else:
        direcao = None
    
    return {
        'convergente': convergente,
        'direcao': direcao,
        'sinal_diario': sinal_diario,
        'sinal_semanal': sinal_semanal
    }


def calcular_entrada_stop_alvo(df, direcao, rr_ratio=2.0):
    """
    Calcula ponto de entrada, stop loss e alvo
    
    Args:
        df: DataFrame com indicador calculado
        direcao: 'COMPRA' ou 'VENDA'
        rr_ratio: Risk/Reward ratio
        
    Returns:
        Dict com entrada, stop e alvo
    """
    ultima_linha = df.iloc[-1]
    
    entrada = ultima_linha['Close']
    
    if direcao == 'COMPRA':
        stop = ultima_linha['linha_inferior']
        distancia = entrada - stop
        alvo = entrada + (distancia * rr_ratio)
    else:  # VENDA
        stop = ultima_linha['linha_superior']
        distancia = stop - entrada
        alvo = entrada - (distancia * rr_ratio)
    
    return {
        'entrada': entrada,
        'stop': stop,
        'alvo': alvo,
        'distancia': distancia,
        'rr': f"1:{rr_ratio}"
    }


# ==========================================
# VISUALIZAÇÕES
# ==========================================

def criar_grafico_cacaus_channel(df, ticker, timeframe="Diário"):
    """
    Cria gráfico do Cacau's Channel
    
    Args:
        df: DataFrame com indicador
        ticker: Nome do ativo
        timeframe: "Diário" ou "Semanal"
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Preço',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Linha Superior (vermelha)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_superior'],
        mode='lines',
        name='Linha Superior',
        line=dict(color='red', width=2)
    ))
    
    # Linha Inferior (verde)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_inferior'],
        mode='lines',
        name='Linha Inferior',
        line=dict(color='lime', width=2)
    ))
    
    # Linha Média (branca)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_media'],
        mode='lines',
        name='Linha Média',
        line=dict(color='white', width=2)
    ))
    
    # EMA da Média (laranja)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema_media'],
        mode='lines',
        name='EMA Média',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Layout
    fig.update_layout(
        title=f"{ticker} - Cacau's Channel ({timeframe})",
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        height=600,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

st.title("🎯 Cacau's Channel")
st.markdown("Análise técnica com convergência multi-timeframe (Diário + Semanal)")

# Painel de cache
try:
    cache_manager.exibir_painel_controle()
except:
    pass

st.markdown("---")


# ==========================================
# SIDEBAR - CONFIGURAÇÕES
# ==========================================

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Teste de email
    st.subheader("📧 Email")
    
    email_configurado = testar_configuracao_email()
    
    if email_configurado:
        st.success("✅ Email configurado")
        
        if st.button("📨 Enviar Email Teste", use_container_width=True):
            with st.spinner("Enviando..."):
                if enviar_email_teste():
                    st.success("✅ Email enviado! Verifique sua caixa de entrada.")
                else:
                    st.error("❌ Erro ao enviar. Verifique os Secrets.")
    else:
        st.error("❌ Email não configurado")
        st.info("Configure em Settings → Secrets")
    
    st.markdown("---")
    
    # Parâmetros do indicador
    st.subheader("📊 Parâmetros")
    
    periodo_superior = st.number_input(
        "Período Superior",
        min_value=5,
        max_value=50,
        value=20,
        step=1
    )
    
    periodo_inferior = st.number_input(
        "Período Inferior",
        min_value=5,
        max_value=50,
        value=30,
        step=1
    )
    
    ema_periodo = st.number_input(
        "EMA Período",
        min_value=3,
        max_value=30,
        value=9,
        step=1
    )
    
    rr_ratio = st.selectbox(
        "Risk/Reward",
        options=[1.5, 2.0, 2.5, 3.0],
        index=1,
        format_func=lambda x: f"1:{x}"
    )
    
    st.markdown("---")
    
    # Período de análise
    st.subheader("📅 Período")
    
    data_fim = st.date_input(
        "Data Final",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    periodo_analise = st.selectbox(
        "Período de Análise",
        options=["3 meses", "6 meses", "1 ano", "2 anos"],
        index=2
    )
    
    periodos_dias = {
        "3 meses": 90,
        "6 meses": 180,
        "1 ano": 365,
        "2 anos": 730
    }
    
    data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=periodos_dias[periodo_analise])
    data_fim_dt = datetime.combine(data_fim, datetime.min.time())


# ==========================================
# SELEÇÃO DE ATIVOS
# ==========================================

st.subheader("📈 Ativos para Análise")

# Opção 1: Usar portfólio salvo
portfolios_disponiveis = []
try:
    from core.portfolio import listar_portfolios, carregar_portfolio
    portfolios_disponiveis = listar_portfolios()
except:
    pass

usar_portfolio = False

if portfolios_disponiveis:
    usar_portfolio = st.checkbox("Usar portfólio salvo", value=False)
    
    if usar_portfolio:
        portfolio_selecionado = st.selectbox(
            "Selecione o portfólio",
            portfolios_disponiveis
        )
        
        portfolio = carregar_portfolio(portfolio_selecionado)
        tickers = portfolio.tickers if portfolio else []
        st.info(f"📊 {len(tickers)} ativos do portfólio '{portfolio_selecionado}'")

if not usar_portfolio or not portfolios_disponiveis:
    # Opção 2: Input manual
    tickers_input = st.text_input(
        "Ativos (separados por vírgula)",
        value="PETR4,VALE3,ITUB4",
        help="Ex: PETR4,VALE3,ITUB4"
    )
    
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    st.info(f"📊 {len(tickers)} ativos selecionados")

if not tickers:
    st.warning("⚠️ Selecione pelo menos um ativo")
    st.stop()

st.markdown("---")


# ==========================================
# BOTÃO DE ANÁLISE
# ==========================================

if st.button("🚀 Analisar Oportunidades", type="primary", use_container_width=True):
    
    oportunidades = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        
        progress = (idx + 1) / len(tickers)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {ticker}... ({idx+1}/{len(tickers)})")
        
        try:
            # Buscar dados
            df = get_price_history([ticker], data_inicio, data_fim_dt)
            
            if df.empty or ticker not in df.columns:
                continue
            
            # Preparar dados
            df_ativo = pd.DataFrame({
                'Open': df[ticker],
                'High': df[ticker],
                'Low': df[ticker],
                'Close': df[ticker],
                'Volume': 0
            })
            
            df_ativo = df_ativo.dropna()
            
            if len(df_ativo) < max(periodo_superior, periodo_inferior, ema_periodo):
                continue
            
            # Calcular indicador no DIÁRIO
            df_diario = calcular_cacaus_channel(
                df_ativo,
                periodo_superior,
                periodo_inferior,
                ema_periodo
            )
            
            # Converter para SEMANAL
            df_semanal_raw = resample_para_semanal(df_ativo)
            
            if len(df_semanal_raw) < max(periodo_superior, periodo_inferior, ema_periodo):
                continue
            
            # Calcular indicador no SEMANAL
            df_semanal = calcular_cacaus_channel(
                df_semanal_raw,
                periodo_superior,
                periodo_inferior,
                ema_periodo
            )
            
            # Detectar convergência
            convergencia = detectar_convergencia(df_diario, df_semanal)
            
            if convergencia['convergente']:
                # Calcular entrada, stop e alvo
                pontos = calcular_entrada_stop_alvo(
                    df_diario,
                    convergencia['direcao'],
                    rr_ratio
                )
                
                oportunidades.append({
                    'ticker': ticker,
                    'direcao': convergencia['direcao'],
                    'entrada': pontos['entrada'],
                    'stop': pontos['stop'],
                    'alvo': pontos['alvo'],
                    'rr': pontos['rr'],
                    'df_diario': df_diario,
                    'df_semanal': df_semanal
                })
        
        except Exception as e:
            st.warning(f"⚠️ Erro ao analisar {ticker}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Salvar oportunidades no session_state
    st.session_state.cacaus_oportunidades = oportunidades
    
    st.markdown("---")
    
    # Mostrar resultados
    if oportunidades:
        st.success(f"✅ {len(oportunidades)} oportunidade(s) detectada(s)!")
        
        # Botão de enviar email
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if email_configurado:
                if st.button("📧 Enviar por Email", use_container_width=True):
                    with st.spinner("Enviando email..."):
                        if enviar_alerta_oportunidades(oportunidades):
                            st.success("✅ Email enviado com sucesso!")
                        else:
                            st.error("❌ Erro ao enviar email")
    else:
        st.info("ℹ️ Nenhuma oportunidade com convergência detectada no momento")


# ==========================================
# EXIBIR OPORTUNIDADES
# ==========================================

if 'cacaus_oportunidades' in st.session_state and st.session_state.cacaus_oportunidades:
    
    oportunidades = st.session_state.cacaus_oportunidades
    
    st.markdown("---")
    st.header("📊 Oportunidades Detectadas")
    
    # Tabela resumo
    st.subheader("📋 Resumo")
    
    df_oportunidades = pd.DataFrame([
        {
            'Ativo': opp['ticker'],
            'Direção': opp['direcao'],
            'Entrada': f"R$ {opp['entrada']:.2f}",
            'Stop Loss': f"R$ {opp['stop']:.2f}",
            'Alvo': f"R$ {opp['alvo']:.2f}",
            'R/R': opp['rr']
        }
        for opp in oportunidades
    ])
    
    st.dataframe(
        df_oportunidades,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Ativo': st.column_config.TextColumn('Ativo', width="small"),
            'Direção': st.column_config.TextColumn('Direção', width="small"),
            'Entrada': st.column_config.TextColumn('Entrada', width="medium"),
            'Stop Loss': st.column_config.TextColumn('Stop Loss', width="medium"),
            'Alvo': st.column_config.TextColumn('Alvo', width="medium"),
            'R/R': st.column_config.TextColumn('R/R', width="small")
        }
    )
    
    st.markdown("---")
    
    # Gráficos detalhados
    st.subheader("📈 Análise Gráfica")
    
    for opp in oportunidades:
        
        with st.expander(f"📊 {opp['ticker']} - {opp['direcao']}", expanded=True):
            
            # Informações
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entrada", f"R$ {opp['entrada']:.2f}")
            
            with col2:
                st.metric("Stop Loss", f"R$ {opp['stop']:.2f}")
            
            with col3:
                st.metric("Alvo", f"R$ {opp['alvo']:.2f}")
            
            with col4:
                st.metric("R/R", opp['rr'])
            
            # Gráficos
            tab_diario, tab_semanal = st.tabs(["📅 Diário", "📆 Semanal"])
            
            with tab_diario:
                fig_diario = criar_grafico_cacaus_channel(
                    opp['df_diario'].tail(100),
                    opp['ticker'],
                    "Diário"
                )
                st.plotly_chart(fig_diario, use_container_width=True)
            
            with tab_semanal:
                fig_semanal = criar_grafico_cacaus_channel(
                    opp['df_semanal'].tail(50),
                    opp['ticker'],
                    "Semanal"
                )
                st.plotly_chart(fig_semanal, use_container_width=True)


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")
st.markdown("""
### 📖 Como funciona o Cacau's Channel?

**Regras de Sinal:**
- 🟢 **COMPRA:** Linha Branca (Média) acima da Linha Laranja (EMA) no Diário E Semanal
- 🔴 **VENDA:** Linha Branca (Média) abaixo da Linha Laranja (EMA) no Diário E Semanal
- ✅ **Convergência:** Ambos timeframes devem estar alinhados

**Gestão de Risco:**
- **Stop Loss COMPRA:** Linha Inferior (verde)
- **Stop Loss VENDA:** Linha Superior (vermelha)
- **Alvo:** Calculado baseado no Risk/Reward selecionado

⚠️ **Aviso:** Este sistema é apenas uma ferramenta de análise. Não constitui recomendação de investimento.
""")
