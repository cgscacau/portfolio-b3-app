"""
Cacau's Channel - Screener Multi-Timeframe
Analisa todos os ativos e mostra apenas oportunidades com convergência
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os

# Importar módulos
from core.data import get_price_history
from core.cache import cache_manager


# ==========================================
# CARREGAR BASE DE ATIVOS
# ==========================================

@st.cache_data
def carregar_base_ativos():
    """Carrega base completa de ativos da B3"""
    try:
        caminho = os.path.join('assets', 'b3_universe.csv')
        df = pd.read_csv(caminho)
        
        # Garantir que tem coluna de ticker
        if 'ticker' in df.columns:
            tickers = df['ticker'].dropna().unique().tolist()
        elif 'symbol' in df.columns:
            tickers = df['symbol'].dropna().unique().tolist()
        else:
            # Pegar primeira coluna
            tickers = df.iloc[:, 0].dropna().unique().tolist()
        
        # Limpar e normalizar
        tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
        
        return sorted(tickers)
    
    except Exception as e:
        st.error(f"Erro ao carregar base de ativos: {str(e)}")
        return []


# ==========================================
# FUNÇÕES DE CÁLCULO DO INDICADOR
# ==========================================

def calcular_cacaus_channel(df, periodo_superior=20, periodo_inferior=30, ema_periodo=9):
    """Calcula o indicador Cacau's Channel"""
    df = df.copy()
    
    df['linha_superior'] = df['Close'].rolling(window=periodo_superior).max()
    df['linha_inferior'] = df['Close'].rolling(window=periodo_inferior).min()
    df['linha_media'] = (df['linha_superior'] + df['linha_inferior']) / 2
    df['ema_media'] = df['linha_media'].ewm(span=ema_periodo, adjust=False).mean()
    
    df['sinal'] = 0
    df.loc[df['linha_media'] > df['ema_media'], 'sinal'] = 1
    df.loc[df['linha_media'] < df['ema_media'], 'sinal'] = -1
    
    return df


def resample_para_semanal(df):
    """Converte dados diários para semanais"""
    df_semanal = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return df_semanal


def detectar_convergencia(df_diario, df_semanal):
    """Detecta convergência entre timeframes"""
    sinal_diario = df_diario['sinal'].iloc[-1] if not df_diario.empty else 0
    sinal_semanal = df_semanal['sinal'].iloc[-1] if not df_semanal.empty else 0
    
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
    """Calcula ponto de entrada, stop loss e alvo"""
    ultima_linha = df.iloc[-1]
    entrada = ultima_linha['Close']
    
    if direcao == 'COMPRA':
        stop = ultima_linha['linha_inferior']
        distancia = entrada - stop
        alvo = entrada + (distancia * rr_ratio)
    else:
        stop = ultima_linha['linha_superior']
        distancia = stop - entrada
        alvo = entrada - (distancia * rr_ratio)
    
    return {
        'entrada': entrada,
        'stop': stop,
        'alvo': alvo,
        'rr': f"1:{rr_ratio}"
    }


# ==========================================
# VISUALIZAÇÃO
# ==========================================

def criar_grafico_cacaus_channel(df_diario, df_semanal, ticker, timeframe_ativo="Diário"):
    """Cria gráfico do Cacau's Channel com alternância de timeframe"""
    
    df = df_diario if timeframe_ativo == "Diário" else df_semanal
    df = df.tail(100 if timeframe_ativo == "Diário" else 50)
    
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
    
    fig.update_layout(
        title=f"{ticker} - Cacau's Channel ({timeframe_ativo})",
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        height=700,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

st.title("🎯 Cacau's Channel - Screener")
st.markdown("Screener automático com convergência multi-timeframe")

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
    
    st.subheader("📊 Parâmetros")
    
    periodo_superior = st.number_input("Período Superior", min_value=5, max_value=50, value=20, step=1)
    periodo_inferior = st.number_input("Período Inferior", min_value=5, max_value=50, value=30, step=1)
    ema_periodo = st.number_input("EMA Período", min_value=3, max_value=30, value=9, step=1)
    rr_ratio = st.selectbox("Risk/Reward", options=[1.5, 2.0, 2.5, 3.0], index=1, format_func=lambda x: f"1:{x}")
    
    st.markdown("---")
    
    st.subheader("📅 Período")
    
    data_fim = st.date_input("Data Final", value=datetime.now(), max_value=datetime.now())
    periodo_analise = st.selectbox("Período de Análise", options=["3 meses", "6 meses", "1 ano", "2 anos"], index=2)
    
    periodos_dias = {"3 meses": 90, "6 meses": 180, "1 ano": 365, "2 anos": 730}
    
    data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=periodos_dias[periodo_analise])
    data_fim_dt = datetime.combine(data_fim, datetime.min.time())


# ==========================================
# SELEÇÃO DE ATIVOS
# ==========================================

st.subheader("📈 Seleção de Ativos")

# Carregar base completa
base_completa = carregar_base_ativos()

if base_completa:
    st.success(f"✅ Base carregada: {len(base_completa)} ativos disponíveis")
else:
    st.warning("⚠️ Não foi possível carregar a base de ativos")

# Opções de seleção
opcao_selecao = st.radio(
    "Como deseja selecionar os ativos?",
    options=[
        "📁 Usar portfólio salvo",
        "🌐 Usar base completa da B3",
        "✍️ Lista personalizada"
    ],
    horizontal=False
)

tickers = []

# OPÇÃO 1: Portfólio salvo
if opcao_selecao == "📁 Usar portfólio salvo":
    try:
        from core.portfolio import listar_portfolios, carregar_portfolio
        portfolios_disponiveis = listar_portfolios()
        
        if portfolios_disponiveis:
            portfolio_selecionado = st.selectbox("Selecione o portfólio", portfolios_disponiveis)
            portfolio = carregar_portfolio(portfolio_selecionado)
            tickers = portfolio.tickers if portfolio else []
            st.info(f"📊 {len(tickers)} ativos do portfólio '{portfolio_selecionado}'")
        else:
            st.warning("Nenhum portfólio salvo encontrado")
    except:
        st.error("Erro ao carregar portfólios")

# OPÇÃO 2: Base completa
elif opcao_selecao == "🌐 Usar base completa da B3":
    if base_completa:
        
        # Filtros
        col1, col2 = st.columns(2)
        
        with col1:
            filtro_tipo = st.multiselect(
                "Filtrar por tipo",
                options=["Ações", "FIIs", "ETFs", "Todos"],
                default=["Todos"]
            )
        
        with col2:
            limite_ativos = st.number_input(
                "Limite de ativos (0 = todos)",
                min_value=0,
                max_value=len(base_completa),
                value=50,
                step=10,
                help="Para evitar sobrecarga, limite a quantidade de ativos"
            )
        
        # Aplicar filtros
        if "Todos" in filtro_tipo:
            tickers = base_completa
        else:
            tickers_filtrados = []
            
            if "Ações" in filtro_tipo:
                # Ações terminam com 3 ou 4
                tickers_filtrados.extend([t for t in base_completa if t[-1] in ['3', '4']])
            
            if "FIIs" in filtro_tipo:
                # FIIs terminam com 11
                tickers_filtrados.extend([t for t in base_completa if t.endswith('11')])
            
            if "ETFs" in filtro_tipo:
                # ETFs geralmente contém 'B' no final
                tickers_filtrados.extend([t for t in base_completa if 'B' in t[-2:]])
            
            tickers = sorted(list(set(tickers_filtrados)))
        
        # Aplicar limite
        if limite_ativos > 0 and len(tickers) > limite_ativos:
            tickers = tickers[:limite_ativos]
            st.warning(f"⚠️ Limitado a {limite_ativos} ativos. Ajuste o limite se quiser analisar mais.")
        
        st.info(f"📊 {len(tickers)} ativos selecionados para análise")
        
        with st.expander("👀 Ver lista de ativos"):
            # Mostrar em colunas
            num_colunas = 5
            cols = st.columns(num_colunas)
            for idx, ticker in enumerate(tickers):
                with cols[idx % num_colunas]:
                    st.text(ticker)
    else:
        st.error("Base de ativos não disponível")

# OPÇÃO 3: Lista personalizada
elif opcao_selecao == "✍️ Lista personalizada":
    tickers_input = st.text_area(
        "Digite os ativos (um por linha ou separados por vírgula)",
        value="PETR4\nVALE3\nITUB4\nBBDC4\nWEGE3",
        height=150
    )
    
    # Processar input (aceita vírgula ou linha)
    tickers_raw = tickers_input.replace(',', '\n').split('\n')
    tickers = [t.strip().upper() for t in tickers_raw if t.strip()]
    
    st.info(f"📊 {len(tickers)} ativos digitados")

# Validação
if not tickers:
    st.warning("⚠️ Selecione pelo menos um ativo para continuar")
    st.stop()

st.markdown("---")


# ==========================================
# ANÁLISE AUTOMÁTICA
# ==========================================

if st.button("🔍 Executar Screener", type="primary", use_container_width=True):
    
    oportunidades = []
    todos_dados = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    contador_text = st.empty()
    
    total_analisados = 0
    total_com_dados = 0
    total_convergentes = 0
    
    for idx, ticker in enumerate(tickers):
        
        progress = (idx + 1) / len(tickers)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {ticker}...")
        contador_text.info(f"📊 Progresso: {idx+1}/{len(tickers)} | ✅ Com dados: {total_com_dados} | 🎯 Convergentes: {total_convergentes}")
        
        total_analisados += 1
        
        try:
            df = get_price_history([ticker], data_inicio, data_fim_dt)
            
            if df.empty or ticker not in df.columns:
                continue
            
            df_ativo = pd.DataFrame({
                'Open': df[ticker],
                'High': df[ticker],
                'Low': df[ticker],
                'Close': df[ticker],
                'Volume': 0
            }).dropna()
            
            if len(df_ativo) < max(periodo_superior, periodo_inferior, ema_periodo):
                continue
            
            total_com_dados += 1
            
            df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
            df_semanal_raw = resample_para_semanal(df_ativo)
            
            if len(df_semanal_raw) < max(periodo_superior, periodo_inferior, ema_periodo):
                continue
            
            df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
            
            convergencia = detectar_convergencia(df_diario, df_semanal)
            
            todos_dados[ticker] = {
                'df_diario': df_diario,
                'df_semanal': df_semanal,
                'convergencia': convergencia
            }
            
            if convergencia['convergente']:
                total_convergentes += 1
                pontos = calcular_entrada_stop_alvo(df_diario, convergencia['direcao'], rr_ratio)
                
                oportunidades.append({
                    'ticker': ticker,
                    'direcao': convergencia['direcao'],
                    'entrada': pontos['entrada'],
                    'stop': pontos['stop'],
                    'alvo': pontos['alvo'],
                    'rr': pontos['rr']
                })
        
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    contador_text.empty()
    
    st.session_state.cacaus_oportunidades = oportunidades
    st.session_state.cacaus_todos_dados = todos_dados
    
    # Resumo da análise
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ativos Analisados", total_analisados)
    
    with col2:
        st.metric("Com Dados Suficientes", total_com_dados)
    
    with col3:
        st.metric("🎯 Convergentes", total_convergentes)
    
    if oportunidades:
        st.success(f"✅ {len(oportunidades)} oportunidade(s) com convergência detectada(s)!")
    else:
        st.info("ℹ️ Nenhuma oportunidade com convergência no momento")


# ==========================================
# SCREENER - TABELA DE OPORTUNIDADES
# ==========================================

if 'cacaus_oportunidades' in st.session_state and st.session_state.cacaus_oportunidades:
    
    oportunidades = st.session_state.cacaus_oportunidades
    
    st.markdown("---")
    st.header("📊 Screener - Oportunidades Detectadas")
    
    df_screener = pd.DataFrame([
        {
            'Ativo': opp['ticker'],
            'Direção': opp['direcao'],
            'Entrada': opp['entrada'],
            'Stop Loss': opp['stop'],
            'Alvo': opp['alvo'],
            'R/R': opp['rr']
        }
        for opp in oportunidades
    ])
    
    st.subheader("🎯 Selecione um ativo para visualizar")
    
    ativo_selecionado = st.selectbox(
        "Ativo",
        options=[opp['ticker'] for opp in oportunidades],
        format_func=lambda x: f"{x} - {next((o['direcao'] for o in oportunidades if o['ticker'] == x), '')}"
    )
    
    with st.expander("📋 Ver Tabela Completa", expanded=False):
        st.dataframe(
            df_screener,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Entrada': st.column_config.NumberColumn('Entrada', format="R$ %.2f"),
                'Stop Loss': st.column_config.NumberColumn('Stop Loss', format="R$ %.2f"),
                'Alvo': st.column_config.NumberColumn('Alvo', format="R$ %.2f")
            }
        )
    
    st.markdown("---")
    
    opp_selecionada = next((o for o in oportunidades if o['ticker'] == ativo_selecionado), None)
    
    if opp_selecionada and ativo_selecionado in st.session_state.cacaus_todos_dados:
        
        dados_ativo = st.session_state.cacaus_todos_dados[ativo_selecionado]
        
        st.header(f"📈 Análise Gráfica")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            direcao_cor = "🟢" if opp_selecionada['direcao'] == 'COMPRA' else "🔴"
            st.metric("Direção", f"{direcao_cor} {opp_selecionada['direcao']}")
        
        with col2:
            st.metric("Entrada", f"R$ {opp_selecionada['entrada']:.2f}")
        
        with col3:
            st.metric("Stop Loss", f"R$ {opp_selecionada['stop']:.2f}")
        
        with col4:
            st.metric("Alvo", f"R$ {opp_selecionada['alvo']:.2f}")
        
        timeframe = st.radio(
            "Timeframe",
            options=["Diário", "Semanal"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        fig = criar_grafico_cacaus_channel(
            dados_ativo['df_diario'],
            dados_ativo['df_semanal'],
            ativo_selecionado,
            timeframe
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif 'cacaus_todos_dados' in st.session_state:
    st.info("ℹ️ Nenhuma oportunidade com convergência detectada. Execute o screener novamente.")


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")
st.markdown("""
### 📖 Como funciona o Screener?

**Processo:**
1. 🔍 Seleciona fonte de ativos (Portfólio, Base B3 completa ou Lista manual)
2. 📊 Calcula o Cacau's Channel em timeframe Diário e Semanal para cada ativo
3. ✅ Identifica apenas ativos com **convergência** entre os dois timeframes
4. 📋 Exibe tabela resumida (screener) apenas com oportunidades
5. 📈 Permite visualizar gráfico de cada ativo individualmente

**Regras de Convergência:**
- 🟢 **COMPRA:** Linha Branca > Linha Laranja (Diário E Semanal)
- 🔴 **VENDA:** Linha Branca < Linha Laranja (Diário E Semanal)

⚠️ **Aviso:** Ferramenta de análise técnica. Não constitui recomendação de investimento.
""")
