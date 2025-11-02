"""
Cacau's Channel - Screener Multi-Timeframe
Analisa todos os ativos e mostra sinais e convergências
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import yfinance as yf

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
        
        if 'ticker' in df.columns:
            tickers = df['ticker'].dropna().unique().tolist()
        elif 'symbol' in df.columns:
            tickers = df['symbol'].dropna().unique().tolist()
        else:
            tickers = df.iloc[:, 0].dropna().unique().tolist()
        
        tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
        
        return sorted(tickers)
    
    except Exception as e:
        return []


# ==========================================
# BUSCAR DADOS OHLC COMPLETOS
# ==========================================

@st.cache_data(ttl=3600)
def buscar_dados_ohlc(ticker, data_inicio, data_fim):
    """
    Busca dados OHLC completos usando yfinance diretamente
    
    Args:
        ticker: Código do ativo
        data_inicio: Data inicial
        data_fim: Data final
        
    Returns:
        DataFrame com OHLC completo
    """
    try:
        ticker_yf = ticker if ticker.endswith('.SA') else f"{ticker}.SA"
        
        df = yf.download(
            ticker_yf,
            start=data_inicio,
            end=data_fim + timedelta(days=1),
            progress=False
        )
        
        if df.empty:
            return None
        
        # Renomear colunas se necessário
        if 'Adj Close' in df.columns:
            df = df.drop('Adj Close', axis=1)
        
        # Garantir nomes corretos
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    except Exception as e:
        return None


# ==========================================
# FUNÇÕES DE CÁLCULO DO INDICADOR
# ==========================================

def calcular_cacaus_channel(df, periodo_superior=20, periodo_inferior=30, ema_periodo=9):
    """Calcula o indicador Cacau's Channel"""
    df = df.copy()
    
    # Usar High para linha superior e Low para linha inferior
    df['linha_superior'] = df['High'].rolling(window=periodo_superior).max()
    df['linha_inferior'] = df['Low'].rolling(window=periodo_inferior).min()
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


def detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback=5):
    """Detecta convergência de CRUZAMENTOS entre timeframes"""
    
    if len(df_diario) < lookback + 1 or len(df_semanal) < lookback + 1:
        return {
            'convergente': False,
            'tem_cruzamento': False,
            'alinhado': False,
            'direcao': None,
            'tipo_sinal': None,
            'barra_cruzamento_diario': None,
            'barra_cruzamento_semanal': None,
            'cruzamento_diario': None,
            'cruzamento_semanal': None,
            'posicao_diario': None,
            'posicao_semanal': None
        }
    
    # Posição atual (alinhamento)
    linha_media_diario = df_diario['linha_media'].iloc[-1]
    ema_media_diario = df_diario['ema_media'].iloc[-1]
    posicao_diario = 'COMPRA' if linha_media_diario > ema_media_diario else 'VENDA'
    
    linha_media_semanal = df_semanal['linha_media'].iloc[-1]
    ema_media_semanal = df_semanal['ema_media'].iloc[-1]
    posicao_semanal = 'COMPRA' if linha_media_semanal > ema_media_semanal else 'VENDA'
    
    # Verificar alinhamento (convergência de posição)
    alinhado = posicao_diario == posicao_semanal
    
    # Detectar cruzamento no DIÁRIO
    cruzamento_diario = None
    barra_cruz_diario = None
    
    for i in range(1, min(lookback + 1, len(df_diario))):
        linha_media_atual = df_diario['linha_media'].iloc[-i]
        ema_media_atual = df_diario['ema_media'].iloc[-i]
        linha_media_anterior = df_diario['linha_media'].iloc[-(i+1)]
        ema_media_anterior = df_diario['ema_media'].iloc[-(i+1)]
        
        if linha_media_anterior <= ema_media_anterior and linha_media_atual > ema_media_atual:
            cruzamento_diario = 'COMPRA'
            barra_cruz_diario = i
            break
        
        if linha_media_anterior >= ema_media_anterior and linha_media_atual < ema_media_atual:
            cruzamento_diario = 'VENDA'
            barra_cruz_diario = i
            break
    
    # Detectar cruzamento no SEMANAL
    cruzamento_semanal = None
    barra_cruz_semanal = None
    
    for i in range(1, min(lookback + 1, len(df_semanal))):
        linha_media_atual = df_semanal['linha_media'].iloc[-i]
        ema_media_atual = df_semanal['ema_media'].iloc[-i]
        linha_media_anterior = df_semanal['linha_media'].iloc[-(i+1)]
        ema_media_anterior = df_semanal['ema_media'].iloc[-(i+1)]
        
        if linha_media_anterior <= ema_media_anterior and linha_media_atual > ema_media_atual:
            cruzamento_semanal = 'COMPRA'
            barra_cruz_semanal = i
            break
        
        if linha_media_anterior >= ema_media_anterior and linha_media_atual < ema_media_atual:
            cruzamento_semanal = 'VENDA'
            barra_cruz_semanal = i
            break
    
    # Verificar se tem cruzamento
    tem_cruzamento = cruzamento_diario is not None and cruzamento_semanal is not None
    
    # Convergência de cruzamento (SINAL)
    convergente_cruzamento = False
    direcao_sinal = None
    tipo_sinal = None
    
    if tem_cruzamento and cruzamento_diario == cruzamento_semanal:
        convergente_cruzamento = True
        direcao_sinal = cruzamento_diario
        
        if barra_cruz_diario == 1 and barra_cruz_semanal == 1:
            tipo_sinal = 'SIMULTÂNEO'
        elif barra_cruz_diario == 1:
            tipo_sinal = 'REENTRADA DIÁRIO'
        elif barra_cruz_semanal == 1:
            tipo_sinal = 'REENTRADA SEMANAL'
        else:
            tipo_sinal = 'RECENTE'
    
    return {
        'convergente': convergente_cruzamento,  # Convergência de cruzamento (SINAL)
        'tem_cruzamento': tem_cruzamento,
        'alinhado': alinhado,  # Convergência de posição (ALINHAMENTO)
        'direcao': direcao_sinal,
        'tipo_sinal': tipo_sinal,
        'barra_cruzamento_diario': barra_cruz_diario,
        'barra_cruzamento_semanal': barra_cruz_semanal,
        'cruzamento_diario': cruzamento_diario,
        'cruzamento_semanal': cruzamento_semanal,
        'posicao_diario': posicao_diario,
        'posicao_semanal': posicao_semanal
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
    """Cria gráfico do Cacau's Channel - PERÍODO COMPLETO"""
    
    df = df_diario if timeframe_ativo == "Diário" else df_semanal
    
    # MOSTRAR TODO O PERÍODO (não limitar)
    
    fig = go.Figure()
    
    # Candlestick com OHLC completo
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Preço',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350'
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
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Ajustar zoom para mostrar todo o período
    fig.update_xaxes(range=[df.index[0], df.index[-1]])
    
    return fig


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

st.title("🎯 Cacau's Channel - Screener")
st.markdown("Screener automático com detecção de cruzamentos e convergências")

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
    lookback_cruzamento = st.number_input("Lookback Cruzamento", min_value=1, max_value=10, value=5, step=1)
    
    st.markdown("---")
    
    st.subheader("📅 Período")
    
    data_fim = st.date_input("Data Final", value=datetime.now(), max_value=datetime.now())
    
    periodo_analise = st.selectbox(
        "Período de Análise",
        options=["3 meses", "6 meses", "1 ano", "2 anos", "3 anos", "5 anos"],
        index=5  # Padrão: 5 anos
    )
    
    periodos_dias = {
        "3 meses": 90,
        "6 meses": 180,
        "1 ano": 365,
        "2 anos": 730,
        "3 anos": 1095,
        "5 anos": 1825
    }
    
    data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=periodos_dias[periodo_analise])
    data_fim_dt = datetime.combine(data_fim, datetime.min.time())


# ==========================================
# LAYOUT EM DUAS COLUNAS
# ==========================================

col_esquerda, col_direita = st.columns([1, 3])


# ==========================================
# COLUNA ESQUERDA: SELEÇÃO E SCREENER
# ==========================================

with col_esquerda:
    
    st.subheader("📈 Ativos")
    
    base_completa = carregar_base_ativos()
    
    if base_completa:
        st.caption(f"✅ {len(base_completa)} ativos")
    
    opcao_selecao = st.radio(
        "Fonte",
        options=["📁 Portfólio", "🌐 Base B3", "✍️ Manual"],
        label_visibility="collapsed"
    )
    
    tickers = []
    
    # OPÇÃO 1: Portfólio
    if opcao_selecao == "📁 Portfólio":
        try:
            from core.portfolio import listar_portfolios, carregar_portfolio
            portfolios_disponiveis = listar_portfolios()
            
            if portfolios_disponiveis:
                portfolio_selecionado = st.selectbox("", portfolios_disponiveis, label_visibility="collapsed")
                portfolio = carregar_portfolio(portfolio_selecionado)
                tickers = portfolio.tickers if portfolio else []
                st.caption(f"📊 {len(tickers)} ativos")
            else:
                st.warning("Sem portfólios")
        except:
            st.error("Erro")
    
    # OPÇÃO 2: Base B3
    elif opcao_selecao == "🌐 Base B3":
        if base_completa:
            
            filtro_tipo = st.multiselect(
                "Tipo",
                options=["Ações", "FIIs", "ETFs", "Todos"],
                default=["Ações"],
                label_visibility="collapsed"
            )
            
            limite_ativos = st.number_input(
                "Limite",
                min_value=0,
                max_value=len(base_completa),
                value=100,
                step=10,
                label_visibility="collapsed"
            )
            
            if "Todos" in filtro_tipo:
                tickers = base_completa
            else:
                tickers_filtrados = []
                
                if "Ações" in filtro_tipo:
                    tickers_filtrados.extend([t for t in base_completa if t[-1] in ['3', '4'] and not t.endswith('11')])
                
                if "FIIs" in filtro_tipo:
                    tickers_filtrados.extend([t for t in base_completa if t.endswith('11')])
                
                if "ETFs" in filtro_tipo:
                    tickers_filtrados.extend([t for t in base_completa if 'B' in t[-2:] and not t[-1].isdigit()])
                
                tickers = sorted(list(set(tickers_filtrados)))
            
            if limite_ativos > 0 and len(tickers) > limite_ativos:
                tickers = tickers[:limite_ativos]
            
            st.caption(f"📊 {len(tickers)} ativos")
    
    # OPÇÃO 3: Manual
    elif opcao_selecao == "✍️ Manual":
        tickers_input = st.text_area(
            "Ativos",
            value="PETR4\nVALE3\nITUB4",
            height=100,
            label_visibility="collapsed"
        )
        
        tickers_raw = tickers_input.replace(',', '\n').split('\n')
        tickers = [t.strip().upper() for t in tickers_raw if t.strip()]
        
        st.caption(f"📊 {len(tickers)} ativos")
    
    st.markdown("---")
    
    # Botão de screener
    if st.button("🔍 Screener", type="primary", use_container_width=True):
        
        sinais = []  # Cruzamentos convergentes
        convergencias = []  # Alinhamentos sem cruzamento
        todos_dados = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_analisados = 0
        total_com_dados = 0
        total_sinais = 0
        total_convergencias = 0
        
        for idx, ticker in enumerate(tickers):
            
            progress = (idx + 1) / len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"{idx+1}/{len(tickers)}")
            
            total_analisados += 1
            
            try:
                # Buscar dados OHLC completos
                df_ativo = buscar_dados_ohlc(ticker, data_inicio, data_fim_dt)
                
                if df_ativo is None or df_ativo.empty:
                    continue
                
                if len(df_ativo) < max(periodo_superior, periodo_inferior, ema_periodo) + 10:
                    continue
                
                total_com_dados += 1
                
                # Calcular indicador no DIÁRIO
                df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
                
                # Converter para SEMANAL
                df_semanal_raw = resample_para_semanal(df_ativo)
                
                if len(df_semanal_raw) < max(periodo_superior, periodo_inferior, ema_periodo) + 2:
                    continue
                
                # Calcular indicador no SEMANAL
                df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                
                # Detectar convergência
                resultado = detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback_cruzamento)
                
                # Salvar TODOS os dados
                todos_dados[ticker] = {
                    'df_diario': df_diario,
                    'df_semanal': df_semanal,
                    'convergencia': resultado
                }
                
                # Classificar em SINAIS ou CONVERGÊNCIAS
                if resultado['convergente']:
                    # TEM CRUZAMENTO CONVERGENTE = SINAL
                    total_sinais += 1
                    pontos = calcular_entrada_stop_alvo(df_diario, resultado['direcao'], rr_ratio)
                    
                    sinais.append({
                        'ticker': ticker,
                        'direcao': resultado['direcao'],
                        'entrada': pontos['entrada'],
                        'stop': pontos['stop'],
                        'alvo': pontos['alvo'],
                        'rr': pontos['rr'],
                        'tipo_sinal': resultado['tipo_sinal']
                    })
                
                elif resultado['alinhado'] and not resultado['tem_cruzamento']:
                    # ALINHADO MAS SEM CRUZAMENTO RECENTE = CONVERGÊNCIA
                    total_convergencias += 1
                    
                    convergencias.append({
                        'ticker': ticker,
                        'direcao': resultado['posicao_diario'],
                        'posicao_diario': resultado['posicao_diario'],
                        'posicao_semanal': resultado['posicao_semanal']
                    })
            
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.cacaus_sinais = sinais
        st.session_state.cacaus_convergencias = convergencias
        st.session_state.cacaus_todos_dados = todos_dados
        
        st.metric("Analisados", total_analisados)
        st.metric("Com Dados", total_com_dados)
        st.metric("🎯 Sinais", total_sinais)
        st.metric("📊 Convergências", total_convergencias)
    
    # Mostrar SINAIS
    st.markdown("---")
    st.subheader("🎯 Sinais")
    
    if 'cacaus_sinais' in st.session_state and st.session_state.cacaus_sinais:
        
        sinais = st.session_state.cacaus_sinais
        
        for sinal in sinais:
            direcao_cor = "🟢" if sinal['direcao'] == 'COMPRA' else "🔴"
            
            if st.button(
                f"{direcao_cor} {sinal['ticker']}",
                key=f"sinal_{sinal['ticker']}",
                use_container_width=True,
                help=f"{sinal['tipo_sinal']}"
            ):
                st.session_state.ativo_visualizar = sinal['ticker']
                st.rerun()
    else:
        st.caption("Nenhum sinal")
    
    # Mostrar CONVERGÊNCIAS
    st.markdown("---")
    st.subheader("📊 Convergências")
    
    if 'cacaus_convergencias' in st.session_state and st.session_state.cacaus_convergencias:
        
        convergencias = st.session_state.cacaus_convergencias
        
        for conv in convergencias:
            direcao_cor = "🟢" if conv['direcao'] == 'COMPRA' else "🔴"
            
            if st.button(
                f"{direcao_cor} {conv['ticker']}",
                key=f"conv_{conv['ticker']}",
                use_container_width=True,
                help="Alinhado sem cruzamento"
            ):
                st.session_state.ativo_visualizar = conv['ticker']
                st.rerun()
    else:
        st.caption("Nenhuma convergência")


# ==========================================
# COLUNA DIREITA: GRÁFICO
# ==========================================

with col_direita:
    
    st.subheader("📈 Gráfico do Indicador")
    
    if 'cacaus_todos_dados' in st.session_state and st.session_state.cacaus_todos_dados:
        
        ativos_disponiveis = sorted(list(st.session_state.cacaus_todos_dados.keys()))
        
        ativo_padrao = st.session_state.get('ativo_visualizar', ativos_disponiveis[0])
        
        if ativo_padrao not in ativos_disponiveis:
            ativo_padrao = ativos_disponiveis[0]
        
        ativo_selecionado = st.selectbox(
            "Ativo",
            options=ativos_disponiveis,
            index=ativos_disponiveis.index(ativo_padrao) if ativo_padrao in ativos_disponiveis else 0
        )
        
        dados_ativo = st.session_state.cacaus_todos_dados[ativo_selecionado]
        
        # Verificar se tem sinal
        tem_sinal = False
        sinal_info = None
        
        if 'cacaus_sinais' in st.session_state:
            sinal_info = next(
                (s for s in st.session_state.cacaus_sinais if s['ticker'] == ativo_selecionado),
                None
            )
            tem_sinal = sinal_info is not None
        
        # Verificar se tem convergência
        tem_convergencia = False
        conv_info = None
        
        if 'cacaus_convergencias' in st.session_state:
            conv_info = next(
                (c for c in st.session_state.cacaus_convergencias if c['ticker'] == ativo_selecionado),
                None
            )
            tem_convergencia = conv_info is not None
        
        # Mostrar status
        if tem_sinal:
            st.success(f"🎯 SINAL: {sinal_info['direcao']}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                direcao_cor = "🟢" if sinal_info['direcao'] == 'COMPRA' else "🔴"
                st.metric("Direção", f"{direcao_cor} {sinal_info['direcao']}")
            
            with col2:
                st.metric("Entrada", f"R$ {sinal_info['entrada']:.2f}")
            
            with col3:
                st.metric("Stop", f"R$ {sinal_info['stop']:.2f}")
            
            with col4:
                st.metric("Alvo", f"R$ {sinal_info['alvo']:.2f}")
            
            with col5:
                st.metric("Tipo", sinal_info['tipo_sinal'])
        
        elif tem_convergencia:
            st.info(f"📊 CONVERGÊNCIA: {conv_info['direcao']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"📅 Diário: {conv_info['posicao_diario']}")
            
            with col2:
                st.write(f"📆 Semanal: {conv_info['posicao_semanal']}")
        
        else:
            conv = dados_ativo['convergencia']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if conv['cruzamento_diario']:
                    cor = "🟢" if conv['cruzamento_diario'] == 'COMPRA' else "🔴"
                    st.info(f"📅 D: {cor}")
                else:
                    st.warning("📅 D: -")
            
            with col2:
                if conv['cruzamento_semanal']:
                    cor = "🟢" if conv['cruzamento_semanal'] == 'COMPRA' else "🔴"
                    st.info(f"📆 S: {cor}")
                else:
                    st.warning("📆 S: -")
            
            with col3:
                if conv['alinhado']:
                    st.success("✅ Alinh.")
                else:
                    st.error("❌ Desalinh.")
        
        # Timeframe
        timeframe = st.radio(
            "Timeframe",
            options=["Diário", "Semanal"],
            horizontal=True
        )
        
        # Gráfico - PERÍODO COMPLETO
        fig = criar_grafico_cacaus_channel(
            dados_ativo['df_diario'],
            dados_ativo['df_semanal'],
            ativo_selecionado,
            timeframe
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Execute o screener")
        
        st.markdown("---")
        st.subheader("🔍 Análise Individual")
        
        ticker_individual = st.text_input("Ticker", value="PETR4")
        
        if st.button("📊 Visualizar", use_container_width=True):
            
            with st.spinner(f"Carregando..."):
                
                try:
                    df_ativo = buscar_dados_ohlc(ticker_individual, data_inicio, data_fim_dt)
                    
                    if df_ativo is not None and not df_ativo.empty:
                        
                        df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
                        df_semanal_raw = resample_para_semanal(df_ativo)
                        df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                        
                        convergencia = detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback_cruzamento)
                        
                        st.session_state.cacaus_todos_dados = {
                            ticker_individual: {
                                'df_diario': df_diario,
                                'df_semanal': df_semanal,
                                'convergencia': convergencia
                            }
                        }
                        
                        st.session_state.ativo_visualizar = ticker_individual
                        st.success("✅ OK!")
                        st.rerun()
                    
                    else:
                        st.error("❌ Sem dados")
                
                except Exception as e:
                    st.error(f"❌ {str(e)}")


# ==========================================
# TABELAS RESUMO (ABAIXO DO LAYOUT)
# ==========================================

st.markdown("---")

# Criar tabs para as tabelas
if 'cacaus_sinais' in st.session_state or 'cacaus_convergencias' in st.session_state:
    
    tab1, tab2 = st.tabs(["🎯 Tabela de Sinais", "📊 Tabela de Convergências"])
    
    with tab1:
        if 'cacaus_sinais' in st.session_state and st.session_state.cacaus_sinais:
            
            sinais = st.session_state.cacaus_sinais
            
            df_sinais = pd.DataFrame([
                {
                    'Ativo': s['ticker'],
                    'Direção': s['direcao'],
                    'Tipo': s['tipo_sinal'],
                    'Entrada': s['entrada'],
                    'Stop': s['stop'],
                    'Alvo': s['alvo'],
                    'R/R': s['rr']
                }
                for s in sinais
            ])
            
            st.dataframe(
                df_sinais,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Entrada': st.column_config.NumberColumn('Entrada', format="R$ %.2f"),
                    'Stop': st.column_config.NumberColumn('Stop', format="R$ %.2f"),
                    'Alvo': st.column_config.NumberColumn('Alvo', format="R$ %.2f")
                }
            )
        else:
            st.info("Nenhum sinal detectado")
    
    with tab2:
        if 'cacaus_convergencias' in st.session_state and st.session_state.cacaus_convergencias:
            
            convergencias = st.session_state.cacaus_convergencias
            
            df_conv = pd.DataFrame([
                {
                    'Ativo': c['ticker'],
                    'Direção': c['direcao'],
                    'Status Diário': c['posicao_diario'],
                    'Status Semanal': c['posicao_semanal']
                }
                for c in convergencias
            ])
            
            st.dataframe(df_conv, use_container_width=True, hide_index=True)
            
            st.info("💡 Estes ativos estão alinhados mas não tiveram cruzamento recente. Monitore para possíveis reentradas.")
        else:
            st.info("Nenhuma convergência sem sinal")


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")

with st.expander("📖 Documentação"):
    st.markdown("""
    ### Diferença entre Sinais e Convergências
    
    **🎯 SINAIS (Cruzamentos Convergentes):**
    - Ocorre cruzamento da Linha Branca com Linha Laranja
    - Cruzamento acontece no Diário E no Semanal
    - Ambos na mesma direção (COMPRA ou VENDA)
    - **Ação:** Entrada no trade
    
    **📊 CONVERGÊNCIAS (Alinhamento):**
    - Ambos timeframes estão posicionados na mesma direção
    - MAS não houve cruzamento recente
    - **Ação:** Monitorar para possível reentrada
    
    **Gestão de Risco:**
    - Stop COMPRA: Linha Inferior
    - Stop VENDA: Linha Superior
    - Alvo: Risk/Reward configurado
    """)
