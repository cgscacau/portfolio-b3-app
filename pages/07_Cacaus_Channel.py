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
        
        if 'ticker' in df.columns:
            tickers = df['ticker'].dropna().unique().tolist()
        elif 'symbol' in df.columns:
            tickers = df['symbol'].dropna().unique().tolist()
        else:
            tickers = df.iloc[:, 0].dropna().unique().tolist()
        
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
    """
    Detecta convergência de CRUZAMENTOS entre timeframes
    
    Args:
        df_diario: DataFrame com indicador no diário
        df_semanal: DataFrame com indicador no semanal
        lookback: Quantas barras olhar para trás para detectar cruzamento
        
    Returns:
        Dict com resultado da convergência
    """
    
    if len(df_diario) < lookback + 1 or len(df_semanal) < lookback + 1:
        return {
            'convergente': False,
            'direcao': None,
            'tipo_sinal': None,
            'barra_cruzamento_diario': None,
            'barra_cruzamento_semanal': None,
            'cruzamento_diario': None,
            'cruzamento_semanal': None
        }
    
    # Detectar cruzamento no DIÁRIO
    cruzamento_diario = None
    barra_cruz_diario = None
    
    for i in range(1, min(lookback + 1, len(df_diario))):
        linha_media_atual = df_diario['linha_media'].iloc[-i]
        ema_media_atual = df_diario['ema_media'].iloc[-i]
        linha_media_anterior = df_diario['linha_media'].iloc[-(i+1)]
        ema_media_anterior = df_diario['ema_media'].iloc[-(i+1)]
        
        # Cruzamento para CIMA (COMPRA)
        if linha_media_anterior <= ema_media_anterior and linha_media_atual > ema_media_atual:
            cruzamento_diario = 'COMPRA'
            barra_cruz_diario = i
            break
        
        # Cruzamento para BAIXO (VENDA)
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
        
        # Cruzamento para CIMA (COMPRA)
        if linha_media_anterior <= ema_media_anterior and linha_media_atual > ema_media_atual:
            cruzamento_semanal = 'COMPRA'
            barra_cruz_semanal = i
            break
        
        # Cruzamento para BAIXO (VENDA)
        if linha_media_anterior >= ema_media_anterior and linha_media_atual < ema_media_atual:
            cruzamento_semanal = 'VENDA'
            barra_cruz_semanal = i
            break
    
    # Verificar convergência de cruzamentos
    convergente = False
    direcao = None
    tipo_sinal = None
    
    if cruzamento_diario and cruzamento_semanal:
        if cruzamento_diario == cruzamento_semanal:
            convergente = True
            direcao = cruzamento_diario
            
            if barra_cruz_diario == 1 and barra_cruz_semanal == 1:
                tipo_sinal = 'SIMULTÂNEO'
            elif barra_cruz_diario == 1:
                tipo_sinal = 'REENTRADA DIÁRIO'
            elif barra_cruz_semanal == 1:
                tipo_sinal = 'REENTRADA SEMANAL'
            else:
                tipo_sinal = 'RECENTE'
    
    return {
        'convergente': convergente,
        'direcao': direcao,
        'tipo_sinal': tipo_sinal,
        'barra_cruzamento_diario': barra_cruz_diario,
        'barra_cruzamento_semanal': barra_cruz_semanal,
        'cruzamento_diario': cruzamento_diario,
        'cruzamento_semanal': cruzamento_semanal
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


def obter_dados_ohlc_completos(ticker, data_inicio, data_fim):
    """
    Obtém dados OHLC completos do ativo usando yfinance diretamente
    para garantir velas completas no gráfico
    """
    try:
        import yfinance as yf
        
        # Adicionar .SA para ativos brasileiros se necessário
        ticker_yf = ticker if '.SA' in ticker else f"{ticker}.SA"
        
        # Baixar dados com yfinance
        dados = yf.download(
            ticker_yf,
            start=data_inicio,
            end=data_fim,
            progress=False,
            auto_adjust=False  # Manter dados originais sem ajuste
        )
        
        if dados.empty:
            return None
        
        # Renomear colunas para padrão
        df_ohlc = pd.DataFrame({
            'Open': dados['Open'],
            'High': dados['High'],
            'Low': dados['Low'],
            'Close': dados['Close'],
            'Volume': dados['Volume']
        }).dropna()
        
        return df_ohlc
        
    except Exception as e:
        # Fallback: usar get_price_history e criar OHLC aproximado
        try:
            df = get_price_history([ticker], data_inicio, data_fim)
            
            if df.empty or ticker not in df.columns:
                return None
            
            # Criar OHLC aproximado
            df_ohlc = pd.DataFrame({
                'Open': df[ticker],
                'High': df[ticker] * 1.001,  # Aproximação: +0.1%
                'Low': df[ticker] * 0.999,   # Aproximação: -0.1%
                'Close': df[ticker],
                'Volume': 0
            }).dropna()
            
            return df_ohlc
            
        except:
            return None


# ==========================================
# VISUALIZAÇÃO
# ==========================================

def criar_grafico_cacaus_channel(df_diario, df_semanal, ticker, timeframe_ativo="Diário"):
    """Cria gráfico do Cacau's Channel com candlesticks completos e melhor centralização"""
    
    df = df_diario if timeframe_ativo == "Diário" else df_semanal
    
    # Mostrar últimas barras
    num_barras = 100 if timeframe_ativo == "Diário" else 50
    df = df.tail(num_barras)
    
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
        decreasing_fillcolor='#ef5350',
        showlegend=True
    ))
    
    # Linha Superior (vermelha)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_superior'],
        mode='lines',
        name='Linha Superior',
        line=dict(color='red', width=2),
        showlegend=True
    ))
    
    # Linha Inferior (verde)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_inferior'],
        mode='lines',
        name='Linha Inferior',
        line=dict(color='lime', width=2),
        showlegend=True
    ))
    
    # Linha Média (branca)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_media'],
        mode='lines',
        name='Linha Média',
        line=dict(color='white', width=2),
        showlegend=True
    ))
    
    # EMA da Média (laranja)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema_media'],
        mode='lines',
        name='EMA Média',
        line=dict(color='orange', width=2, dash='dash'),
        showlegend=True
    ))
    
    # Configuração do layout com melhor centralização
    fig.update_layout(
        title={
            'text': f"{ticker} - Cacau's Channel ({timeframe_ativo})",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        height=700,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        # Melhor centralização e espaçamento
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        # Ajustar eixo Y para melhor visualização
        yaxis=dict(
            autorange=True,
            fixedrange=False
        ),
        # Ajustar eixo X
        xaxis=dict(
            autorange=True,
            rangeslider=dict(visible=False),
            type='date'
        )
    )
    
    # Configurar hover
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128, 128, 128, 0.2)')
    
    return fig


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

st.title("🎯 Cacau's Channel - Screener")
st.markdown("Screener automático com detecção de cruzamentos e convergência")

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
    lookback_cruzamento = st.number_input("Lookback Cruzamento", min_value=1, max_value=10, value=5, step=1, help="Quantas barras olhar para trás")
    
    st.markdown("---")
    
    st.subheader("📅 Período")
    
    data_fim = st.date_input("Data Final", value=datetime.now(), max_value=datetime.now())
    
    # Período expandido para até 5 anos
    periodo_analise = st.selectbox(
        "Período de Análise",
        options=["3 meses", "6 meses", "1 ano", "2 anos", "3 anos", "5 anos"],
        index=2
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
    
    st.caption(f"📅 De {data_inicio.strftime('%d/%m/%Y')} até {data_fim_dt.strftime('%d/%m/%Y')}")


# ==========================================
# LAYOUT EM DUAS COLUNAS
# ==========================================

col_esquerda, col_direita = st.columns([1, 3])


# ==========================================
# COLUNA ESQUERDA: SELEÇÃO E SCREENER
# ==========================================

with col_esquerda:
    
    st.subheader("📈 Ativos")
    
    # Carregar base completa
    base_completa = carregar_base_ativos()
    
    if base_completa:
        st.caption(f"✅ {len(base_completa)} ativos")
    
    # Opções de seleção
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
            st.error("Erro ao carregar")
    
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
                value=50,
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
    
    # Botão de screener
    st.markdown("---")
    
    if st.button("🔍 Screener", type="primary", use_container_width=True):
        
        oportunidades = []
        todos_dados = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_analisados = 0
        total_com_dados = 0
        total_convergentes = 0
        
        for idx, ticker in enumerate(tickers):
            
            progress = (idx + 1) / len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"{idx+1}/{len(tickers)}")
            
            total_analisados += 1
            
            try:
                # Obter dados OHLC completos
                df_ativo = obter_dados_ohlc_completos(ticker, data_inicio, data_fim_dt)
                
                if df_ativo is None or len(df_ativo) < max(periodo_superior, periodo_inferior, ema_periodo) + 10:
                    continue
                
                total_com_dados += 1
                
                df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
                df_semanal_raw = resample_para_semanal(df_ativo)
                
                if len(df_semanal_raw) < max(periodo_superior, periodo_inferior, ema_periodo) + 2:
                    continue
                
                df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                
                convergencia = detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback_cruzamento)
                
                # Salvar TODOS os dados
                todos_dados[ticker] = {
                    'df_diario': df_diario,
                    'df_semanal': df_semanal,
                    'convergencia': convergencia
                }
                
                # Adicionar apenas convergentes
                if convergencia['convergente']:
                    total_convergentes += 1
                    pontos = calcular_entrada_stop_alvo(df_diario, convergencia['direcao'], rr_ratio)
                    
                    oportunidades.append({
                        'ticker': ticker,
                        'direcao': convergencia['direcao'],
                        'entrada': pontos['entrada'],
                        'stop': pontos['stop'],
                        'alvo': pontos['alvo'],
                        'rr': pontos['rr'],
                        'tipo_sinal': convergencia['tipo_sinal']
                    })
            
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.cacaus_oportunidades = oportunidades
        st.session_state.cacaus_todos_dados = todos_dados
        
        st.metric("Analisados", total_analisados)
        st.metric("Com Dados", total_com_dados)
        st.metric("🎯 Sinais", total_convergentes)
        
        if oportunidades:
            st.success(f"✅ {len(oportunidades)} sinal(is)!")
        else:
            st.info("Nenhum sinal")
    
    # Mostrar screener
    st.markdown("---")
    st.subheader("🎯 Sinais")
    
    if 'cacaus_oportunidades' in st.session_state and st.session_state.cacaus_oportunidades:
        
        oportunidades = st.session_state.cacaus_oportunidades
        
        for opp in oportunidades:
            direcao_cor = "🟢" if opp['direcao'] == 'COMPRA' else "🔴"
            
            if st.button(
                f"{direcao_cor} {opp['ticker']}",
                key=f"btn_{opp['ticker']}",
                use_container_width=True,
                help=f"{opp['tipo_sinal']}"
            ):
                st.session_state.ativo_visualizar = opp['ticker']
                st.rerun()
    else:
        st.caption("Execute o screener")


# ==========================================
# COLUNA DIREITA: GRÁFICO
# ==========================================

with col_direita:
    
    st.subheader("📈 Gráfico do Indicador")
    
    # Sempre mostrar gráfico se houver dados
    if 'cacaus_todos_dados' in st.session_state and st.session_state.cacaus_todos_dados:
        
        ativos_disponiveis = sorted(list(st.session_state.cacaus_todos_dados.keys()))
        
        # Usar ativo do session_state ou primeiro da lista
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
        opp_selecionada = None
        if 'cacaus_oportunidades' in st.session_state:
            opp_selecionada = next(
                (o for o in st.session_state.cacaus_oportunidades if o['ticker'] == ativo_selecionado),
                None
            )
        
        # Mostrar informações
        if opp_selecionada:
            # TEM SINAL
            st.success(f"🎯 SINAL: {opp_selecionada['direcao']}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                direcao_cor = "🟢" if opp_selecionada['direcao'] == 'COMPRA' else "🔴"
                st.metric("Direção", f"{direcao_cor} {opp_selecionada['direcao']}")
            
            with col2:
                st.metric("Entrada", f"R$ {opp_selecionada['entrada']:.2f}")
            
            with col3:
                st.metric("Stop", f"R$ {opp_selecionada['stop']:.2f}")
            
            with col4:
                st.metric("Alvo", f"R$ {opp_selecionada['alvo']:.2f}")
            
            with col5:
                st.metric("Tipo", opp_selecionada['tipo_sinal'])
        
        else:
            # SEM SINAL
            conv = dados_ativo['convergencia']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if conv['cruzamento_diario']:
                    cor = "🟢" if conv['cruzamento_diario'] == 'COMPRA' else "🔴"
                    st.info(f"📅 Diário: {cor} {conv['cruzamento_diario']}")
                else:
                    st.warning("📅 Diário: Sem cruz.")
            
            with col2:
                if conv['cruzamento_semanal']:
                    cor = "🟢" if conv['cruzamento_semanal'] == 'COMPRA' else "🔴"
                    st.info(f"📆 Semanal: {cor} {conv['cruzamento_semanal']}")
                else:
                    st.warning("📆 Semanal: Sem cruz.")
            
            with col3:
                if conv['convergente']:
                    st.success("✅ Convergente")
                else:
                    st.error("❌ Sem convergência")
        
        # Timeframe
        timeframe = st.radio(
            "Timeframe",
            options=["Diário", "Semanal"],
            horizontal=True
        )
        
        # Gráfico com velas completas
        fig = criar_grafico_cacaus_channel(
            dados_ativo['df_diario'],
            dados_ativo['df_semanal'],
            ativo_selecionado,
            timeframe
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Execute o screener para visualizar")
        
        # Permitir visualizar ativo individual sem screener
        st.markdown("---")
        st.subheader("🔍 Análise Individual")
        
        ticker_individual = st.text_input("Ticker", value="PETR4")
        
        if st.button("📊 Visualizar", use_container_width=True):
            
            with st.spinner(f"Carregando {ticker_individual}..."):
                
                try:
                    # Obter dados OHLC completos
                    df_ativo = obter_dados_ohlc_completos(ticker_individual, data_inicio, data_fim_dt)
                    
                    if df_ativo is not None and len(df_ativo) >= max(periodo_superior, periodo_inferior, ema_periodo) + 10:
                        
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
                        
                        st.success("✅ Carregado!")
                        st.rerun()
                    
                    else:
                        st.error("❌ Sem dados suficientes")
                
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")

with st.expander("📖 Como funciona?"):
    st.markdown("""
    ### Lógica de Sin<span class="cursor">█</span>
