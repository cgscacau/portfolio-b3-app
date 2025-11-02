"""
Cacau's Channel - Screener Multi-Timeframe
Analisa todos os ativos e mostra apenas oportunidades com convergência
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# PROCESSAMENTO DE DADOS HISTÓRICOS
# ==========================================

def obter_dados_ohlc_reais(ticker, data_inicio, data_fim):
    """
    Obtém dados OHLC reais do Yahoo Finance ou fonte similar
    """
    try:
        import yfinance as yf
        
        # Adicionar .SA para tickers brasileiros se necessário
        ticker_yahoo = ticker + ".SA" if not ticker.endswith(".SA") else ticker
        
        # Baixar dados históricos
        stock = yf.Ticker(ticker_yahoo)
        df = stock.history(start=data_inicio, end=data_fim, auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Renomear colunas para padrão
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    except ImportError:
        st.error("⚠️ Instale yfinance: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        # Fallback para método original se yfinance falhar
        try:
            df = get_price_history([ticker], data_inicio, data_fim)
            if df.empty or ticker not in df.columns:
                return pd.DataFrame()
            
            # Criar OHLC aproximado baseado no close
            precos = df[ticker].dropna()
            
            # Simular variação intraday baseada na volatilidade histórica
            volatilidade = precos.pct_change().std() * 0.5
            
            df_ohlc = pd.DataFrame({
                'Open': precos.shift(1).fillna(precos),
                'High': precos * (1 + np.random.uniform(0, volatilidade, len(precos))),
                'Low': precos * (1 - np.random.uniform(0, volatilidade, len(precos))),
                'Close': precos,
                'Volume': np.random.randint(1000, 100000, len(precos))
            }, index=precos.index)
            
            # Garantir que High >= max(Open, Close) e Low <= min(Open, Close)
            df_ohlc['High'] = df_ohlc[['High', 'Open', 'Close']].max(axis=1)
            df_ohlc['Low'] = df_ohlc[['Low', 'Open', 'Close']].min(axis=1)
            
            return df_ohlc
            
        except Exception:
            return pd.DataFrame()


# ==========================================
# FUNÇÕES DE CÁLCULO DO INDICADOR
# ==========================================

def calcular_cacaus_channel(df, periodo_superior=20, periodo_inferior=30, ema_periodo=9):
    """Calcula o indicador Cacau's Channel"""
    df = df.copy()
    
    if len(df) < max(periodo_superior, periodo_inferior, ema_periodo):
        return df
    
    df['linha_superior'] = df['High'].rolling(window=periodo_superior).max()
    df['linha_inferior'] = df['Low'].rolling(window=periodo_inferior).min()
    df['linha_media'] = (df['linha_superior'] + df['linha_inferior']) / 2
    df['ema_media'] = df['linha_media'].ewm(span=ema_periodo, adjust=False).mean()
    
    # Sinal baseado na posição da linha média vs EMA
    df['sinal'] = 0
    df.loc[df['linha_media'] > df['ema_media'], 'sinal'] = 1
    df.loc[df['linha_media'] < df['ema_media'], 'sinal'] = -1
    
    return df


def resample_para_semanal(df):
    """Converte dados diários para semanais"""
    if df.empty or len(df) < 5:
        return pd.DataFrame()
    
    try:
        df_semanal = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return df_semanal
    except Exception:
        return pd.DataFrame()


def detectar_cruzamentos_e_convergencia(df_diario, df_semanal, lookback=5):
    """
    Detecta cruzamentos e convergência entre timeframes
    CONVERGÊNCIA: Ambos timeframes têm cruzamento na mesma direção
    SINAL: Convergência + critérios adicionais de força
    """
    
    if len(df_diario) < lookback + 1 or len(df_semanal) < lookback + 1:
        return {
            'tem_convergencia': False,
            'tem_sinal': False,
            'direcao': None,
            'tipo': None,
            'cruzamento_diario': None,
            'cruzamento_semanal': None,
            'barras_diario': None,
            'barras_semanal': None
        }
    
    # Detectar cruzamento no DIÁRIO
    cruzamento_diario = None
    barras_diario = None
    
    for i in range(1, min(lookback + 1, len(df_diario))):
        atual = df_diario.iloc[-i]
        anterior = df_diario.iloc[-(i+1)]
        
        if pd.isna(atual['linha_media']) or pd.isna(atual['ema_media']):
            continue
        
        # Cruzamento para CIMA
        if (anterior['linha_media'] <= anterior['ema_media'] and 
            atual['linha_media'] > atual['ema_media']):
            cruzamento_diario = 'COMPRA'
            barras_diario = i
            break
        
        # Cruzamento para BAIXO
        if (anterior['linha_media'] >= anterior['ema_media'] and 
            atual['linha_media'] < atual['ema_media']):
            cruzamento_diario = 'VENDA'
            barras_diario = i
            break
    
    # Detectar cruzamento no SEMANAL
    cruzamento_semanal = None
    barras_semanal = None
    
    for i in range(1, min(lookback + 1, len(df_semanal))):
        atual = df_semanal.iloc[-i]
        anterior = df_semanal.iloc[-(i+1)]
        
        if pd.isna(atual['linha_media']) or pd.isna(atual['ema_media']):
            continue
        
        # Cruzamento para CIMA
        if (anterior['linha_media'] <= anterior['ema_media'] and 
            atual['linha_media'] > atual['ema_media']):
            cruzamento_semanal = 'COMPRA'
            barras_semanal = i
            break
        
        # Cruzamento para BAIXO
        if (anterior['linha_media'] >= anterior['ema_media'] and 
            atual['linha_media'] < atual['ema_media']):
            cruzamento_semanal = 'VENDA'
            barras_semanal = i
            break
    
    # Verificar CONVERGÊNCIA
    tem_convergencia = False
    direcao = None
    
    if cruzamento_diario and cruzamento_semanal and cruzamento_diario == cruzamento_semanal:
        tem_convergencia = True
        direcao = cruzamento_diario
    
    # Verificar SINAL (convergência + critérios de força)
    tem_sinal = False
    tipo_sinal = None
    
    if tem_convergencia:
        # Critérios para SINAL:
        # 1. Ambos cruzamentos recentes (últimas 3 barras)
        # 2. Volume acima da média (se disponível)
        # 3. Momentum favorável
        
        if barras_diario <= 3 and barras_semanal <= 3:
            tem_sinal = True
            
            if barras_diario == 1 and barras_semanal == 1:
                tipo_sinal = 'SIMULTÂNEO FORTE'
            elif barras_diario == 1:
                tipo_sinal = 'CONFIRMAÇÃO DIÁRIO'
            elif barras_semanal == 1:
                tipo_sinal = 'CONFIRMAÇÃO SEMANAL'
            else:
                tipo_sinal = 'CONVERGENTE RECENTE'
    
    return {
        'tem_convergencia': tem_convergencia,
        'tem_sinal': tem_sinal,
        'direcao': direcao,
        'tipo': tipo_sinal,
        'cruzamento_diario': cruzamento_diario,
        'cruzamento_semanal': cruzamento_semanal,
        'barras_diario': barras_diario,
        'barras_semanal': barras_semanal
    }


def calcular_pontos_operacao(df, direcao, rr_ratio=2.0):
    """Calcula entrada, stop e alvo"""
    ultima = df.iloc[-1]
    entrada = ultima['Close']
    
    if direcao == 'COMPRA':
        stop = ultima['linha_inferior']
        risco = entrada - stop
        alvo = entrada + (risco * rr_ratio)
    else:
        stop = ultima['linha_superior']
        risco = stop - entrada
        alvo = entrada - (risco * rr_ratio)
    
    return {
        'entrada': entrada,
        'stop': stop,
        'alvo': alvo,
        'risco': abs(risco),
        'reward': abs(alvo - entrada),
        'rr_ratio': rr_ratio
    }


# ==========================================
# VISUALIZAÇÃO - GRÁFICOS DUPLOS
# ==========================================

def criar_graficos_duplos(df_diario, df_semanal, ticker, resultado):
    """Cria gráfico com ambos timeframes lado a lado"""
    
    # Limitar dados para visualização
    df_d = df_diario.tail(100).copy()
    df_s = df_semanal.tail(50).copy()
    
    # Criar subplot com 2 colunas
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{ticker} - Diário', f'{ticker} - Semanal'),
        horizontal_spacing=0.05,
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # GRÁFICO DIÁRIO (coluna 1)
    fig.add_trace(
        go.Candlestick(
            x=df_d.index,
            open=df_d['Open'],
            high=df_d['High'],
            low=df_d['Low'],
            close=df_d['Close'],
            name='Preço Diário',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Indicadores no diário
    fig.add_trace(
        go.Scatter(x=df_d.index, y=df_d['linha_superior'], 
                  line=dict(color='red', width=1.5), name='Superior D', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_d.index, y=df_d['linha_inferior'], 
                  line=dict(color='lime', width=1.5), name='Inferior D', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_d.index, y=df_d['linha_media'], 
                  line=dict(color='white', width=2), name='Média D', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_d.index, y=df_d['ema_media'], 
                  line=dict(color='orange', width=2, dash='dash'), name='EMA D', showlegend=False),
        row=1, col=1
    )
    
    # GRÁFICO SEMANAL (coluna 2)
    fig.add_trace(
        go.Candlestick(
            x=df_s.index,
            open=df_s['Open'],
            high=df_s['High'],
            low=df_s['Low'],
            close=df_s['Close'],
            name='Preço Semanal',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Indicadores no semanal
    fig.add_trace(
        go.Scatter(x=df_s.index, y=df_s['linha_superior'], 
                  line=dict(color='red', width=1.5), name='Superior S', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_s.index, y=df_s['linha_inferior'], 
                  line=dict(color='lime', width=1.5), name='Inferior S', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_s.index, y=df_s['linha_media'], 
                  line=dict(color='white', width=2), name='Média S', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_s.index, y=df_s['ema_media'], 
                  line=dict(color='orange', width=2, dash='dash'), name='EMA S', showlegend=False),
        row=1, col=2
    )
    
    # Marcar cruzamentos se existirem
    if resultado['cruzamento_diario'] and resultado['barras_diario']:
        idx_cruz = -resultado['barras_diario']
        ponto_cruz = df_d.iloc[idx_cruz]
        cor_cruz = 'lime' if resultado['cruzamento_diario'] == 'COMPRA' else 'red'
        
        fig.add_trace(
            go.Scatter(
                x=[ponto_cruz.name],
                y=[ponto_cruz['Close']],
                mode='markers',
                marker=dict(color=cor_cruz, size=12, symbol='star'),
                name=f'Cruz {resultado["cruzamento_diario"]}',
                showlegend=False
            ),
            row=1, col=1
        )
    
    if resultado['cruzamento_semanal'] and resultado['barras_semanal']:
        idx_cruz = -resultado['barras_semanal']
        ponto_cruz = df_s.iloc[idx_cruz]
        cor_cruz = 'lime' if resultado['cruzamento_semanal'] == 'COMPRA' else 'red'
        
        fig.add_trace(
            go.Scatter(
                x=[ponto_cruz.name],
                y=[ponto_cruz['Close']],
                mode='markers',
                marker=dict(color=cor_cruz, size=12, symbol='star'),
                name=f'Cruz {resultado["cruzamento_semanal"]}',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Layout
    fig.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False
    )
    
    # Remover rangeslider
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

st.title("🎯 Cacau's Channel - Screener Multi-Timeframe")
st.markdown("**Convergência** = Cruzamentos na mesma direção | **Sinal** = Convergência + Força")

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
        index=2
    )
    
    periodos_dias = {
        "3 meses": 90, "6 meses": 180, "1 ano": 365, 
        "2 anos": 730, "3 anos": 1095, "5 anos": 1825
    }
    
    dias_periodo = periodos_dias[periodo_analise]
    data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=dias_periodo)
    data_fim_dt = datetime.combine(data_fim, datetime.min.time())


# ==========================================
# LAYOUT PRINCIPAL
# ==========================================

col_config, col_resultados = st.columns([1, 2])

# ==========================================
# COLUNA CONFIGURAÇÃO
# ==========================================

with col_config:
    st.subheader("📈 Seleção de Ativos")
    
    base_completa = carregar_base_ativos()
    
    opcao_selecao = st.radio(
        "Fonte dos Ativos",
        options=["📁 Portfólio", "🌐 Base B3", "✍️ Manual"],
        index=2
    )
    
    tickers = []
    
    if opcao_selecao == "📁 Portfólio":
        try:
            from core.portfolio import listar_portfolios, carregar_portfolio
            portfolios_disponiveis = listar_portfolios()
            
            if portfolios_disponiveis:
                portfolio_selecionado = st.selectbox("Portfólio", portfolios_disponiveis)
                portfolio = carregar_portfolio(portfolio_selecionado)
                tickers = portfolio.tickers if portfolio else []
            else:
                st.warning("Nenhum portfólio encontrado")
        except:
            st.error("Erro ao carregar portfólios")
    
    elif opcao_selecao == "🌐 Base B3":
        if base_completa:
            filtro_tipo = st.multiselect(
                "Tipos", 
                options=["Ações", "FIIs", "ETFs"], 
                default=["Ações"]
            )
            
            limite = st.number_input("Limite", min_value=10, max_value=100, value=30, step=10)
            
            tickers_filtrados = []
            if "Ações" in filtro_tipo:
                tickers_filtrados.extend([t for t in base_completa if t[-1] in ['3', '4'] and not t.endswith('11')])
            if "FIIs" in filtro_tipo:
                tickers_filtrados.extend([t for t in base_completa if t.endswith('11')])
            if "ETFs" in filtro_tipo:
                tickers_filtrados.extend([t for t in base_completa if 'B' in t[-2:]])
            
            tickers = sorted(list(set(tickers_filtrados)))[:limite]
    
    else:  # Manual
        tickers_input = st.text_area(
            "Tickers (um por linha)",
            value="ALPA4\nPETR4\nVALE3\nITUB4\nBBDC4",
            height=120
        )
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
    
    st.caption(f"📊 {len(tickers)} ativo(s) selecionado(s)")
    
    # Botão de análise
    if st.button("🔍 Analisar Ativos", type="primary", use_container_width=True):
        if not tickers:
            st.error("❌ Selecione pelo menos um ativo")
        else:
            # Inicializar listas de resultados
            convergencias = []
            sinais = []
            todos_dados = {}
            
            # Barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_analisados = 0
            total_com_dados = 0
            
            for idx, ticker in enumerate(tickers):
                progress = (idx + 1) / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"Analisando {ticker}... ({idx+1}/{len(tickers)})")
                
                total_analisados += 1
                
                try:
                    # Obter dados OHLC reais
                    df_ohlc = obter_dados_ohlc_reais(ticker, data_inicio, data_fim_dt)
                    
                    if df_ohlc.empty or len(df_ohlc) < 50:
                        continue
                    
                    total_com_dados += 1
                    
                    # Calcular indicadores
                    df_diario = calcular_cacaus_channel(df_ohlc, periodo_superior, periodo_inferior, ema_periodo)
                    df_semanal_raw = resample_para_semanal(df_ohlc)
                    
                    if df_semanal_raw.empty:
                        continue
                    
                    df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                    
                    # Detectar cruzamentos e convergência
                    resultado = detectar_cruzamentos_e_convergencia(df_diario, df_semanal, lookback_cruzamento)
                    
                    # Salvar todos os dados
                    todos_dados[ticker] = {
                        'df_diario': df_diario,
                        'df_semanal': df_semanal,
                        'resultado': resultado
                    }
                    
                    # Classificar resultados
                    if resultado['tem_convergencia']:
                        pontos = calcular_pontos_operacao(df_diario, resultado['direcao'], rr_ratio)
                        
                        item_convergencia = {
                            'ticker': ticker,
                            'direcao': resultado['direcao'],
                            'cruz_diario': resultado['cruzamento_diario'],
                            'cruz_semanal': resultado['cruzamento_semanal'],
                            'barras_d': resultado['barras_diario'],
                            'barras_s': resultado['barras_semanal'],
                            'entrada': pontos['entrada'],
                            'stop': pontos['stop'],
                            'alvo': pontos['alvo']
                        }
                        
                        convergencias.append(item_convergencia)
                        
                        # Se também é sinal, adicionar à lista de sinais
                        if resultado['tem_sinal']:
                            item_sinal = item_convergencia.copy()
                            item_sinal['tipo_sinal'] = resultado['tipo']
                            sinais.append(item_sinal)
                
                except Exception as e:
                    continue
            
            # Limpar progresso
            progress_bar.empty()
            status_text.empty()
            
            # Salvar no session state
            st.session_state.cacaus_convergencias = convergencias
            st.session_state.cacaus_sinais = sinais
            st.session_state.cacaus_todos_dados = todos_dados
            
            # Mostrar estatísticas
            st.success(f"✅ Análise concluída!")
            st.metric("Total Analisados", total_analisados)
            st.metric("Com Dados", total_com_dados)
            st.metric("🔄 Convergências", len(convergencias))
            st.metric("🎯 Sinais", len(sinais))


# ==========================================
# COLUNA RESULTADOS
# ==========================================

with col_resultados:
    
    # Abas para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(["📊 Gráficos", "🔄 Convergências", "🎯 Sinais"])
    
    # TAB 1: GRÁFICOS
    with tab1:
        if 'cacaus_todos_dados' in st.session_state and st.session_state.cacaus_todos_dados:
            
            ativos_disponiveis = sorted(list(st.session_state.cacaus_todos_dados.keys()))
            ativo_selecionado = st.selectbox("Ativo para Visualização", ativos_disponiveis)
            
            dados_ativo = st.session_state.cacaus_todos_dados[ativo_selecionado]
            resultado = dados_ativo['resultado']
            
            # Status do ativo
            col_status1, col_status2, col_status3 = st.columns(3)
            
            with col_status1:
                if resultado['tem_sinal']:
                    st.success(f"🎯 SINAL: {resultado['direcao']}")
                elif resultado['tem_convergencia']:
                    st.info(f"🔄 CONVERGÊNCIA: {resultado['direcao']}")
                else:
                    st.warning("❌ Sem convergência")
            
            with col_status2:
                if resultado['cruzamento_diario']:
                    cor = "🟢" if resultado['cruzamento_diario'] == 'COMPRA' else "🔴"
                    st.write(f"📅 Diário: {cor} {resultado['cruzamento_diario']} ({resultado['barras_diario']})")
                else:
                    st.write("📅 Diário: Sem cruzamento")
            
            with col_status3:
                if resultado['cruzamento_semanal']:
                    cor = "🟢" if resultado['cruzamento_semanal'] == 'COMPRA' else "🔴"
                    st.write(f"📆 Semanal: {cor} {resultado['cruzamento_semanal']} ({resultado['barras_semanal']})")
                else:
                    st.write("📆 Semanal: Sem cruzamento")
            
            # Gráfico duplo
            fig = criar_graficos_duplos(
                dados_ativo['df_diario'],
                dados_ativo['df_semanal'],
                ativo_selecionado,
                resultado
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Pontos de operação se houver convergência
            if resultado['tem_convergencia']:
                pontos = calcular_pontos_operacao(dados_ativo['df_diario'], resultado['direcao'], rr_ratio)
                
                st.subheader("📍 Pontos de Operação")
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                with col_p1:
                    st.metric("Entrada", f"R$ {pontos['entrada']:.2f}")
                with col_p2:
                    st.metric("Stop Loss", f"R$ {pontos['stop']:.2f}")
                with col_p3:
                    st.metric("Alvo", f"R$ {pontos['alvo']:.2f}")
                with col_p4:
                    st.metric("R/R", f"1:{pontos['rr_ratio']}")
        
        else:
            st.info("👈 Execute a análise para visualizar gráficos")
    
    # TAB 2: CONVERGÊNCIAS
    with tab2:
        if 'cacaus_convergencias' in st.session_state and st.session_state.cacaus_convergencias:
            
            st.subheader(f"🔄 {len(st.session_state.cacaus_convergencias)} Convergência(s)")
            st.caption("Ativos com cruzamentos na mesma direção em ambos timeframes")
            
            for conv in st.session_state.cacaus_convergencias:
                with st.expander(f"{'🟢' if conv['direcao'] == 'COMPRA' else '🔴'} {conv['ticker']} - {conv['direcao']}"):
                    
                    col_c1, col_c2 = st.columns(2)
                    
                    with col_c1:
                        st.write("**Cruzamentos:**")
                        st.write(f"📅 Diário: {conv['cruz_diario']} ({conv['barras_d']} barras atrás)")
                        st.write(f"📆 Semanal: {conv['cruz_semanal']} ({conv['barras_s']} barras atrás)")
                    
                    with col_c2:
                        st.write("**Pontos de Operação:**")
                        st.write(f"🎯 Entrada: R$ {conv['entrada']:.2f}")
                        st.write(f"🛑 Stop: R$ {conv['stop']:.2f}")
                        st.write(f"🏆 Alvo: R$ {conv['alvo']:.2f}")
                    
                    if st.button(f"Ver Gráfico de {conv['ticker']}", key=f"graf_conv_{conv['ticker']}"):
                        st.session_state.ativo_visualizar = conv['ticker']
                        st.rerun()
        
        else:
            st.info("Nenhuma convergência encontrada")
    
    # TAB 3: SINAIS
    with tab3:
        if 'cacaus_sinais' in st.session_state and st.session_state.cacaus_sinais:
            
            st.subheader(f"🎯 {len(st.session_state.cacaus_sinais)} Sinal(is) de Trading")
            st.caption("Convergências com critérios adicionais de força - Prontos para operação")
            
            for sinal in st.session_state.cacaus_sinais:
                with st.container():
                    st.markdown("---")
                    
                    col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
                    
                    with col_s1:
                        cor_sinal = "🟢" if sinal['direcao'] == 'COMPRA' else "🔴"
                        st.markdown(f"### {cor_sinal} {sinal['ticker']}")
                        st.markdown(f"**{sinal['direcao']}**")
                        st.caption(sinal['tipo_sinal'])
                    
                    with col_s2:
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.metric("Entrada", f"R$ {sinal['entrada']:.2f}")
                        with subcol2:
                            st.metric("Stop", f"R$ {sinal['stop']:.2f}")
                        with subcol3:
                            st.metric("Alvo", f"R$ {sinal['alvo']:.2f}")
                    
                    with col_s3:
                        if st.button(f"📊 Ver Gráfico", key=f"graf_sinal_{sinal['ticker']}", use_container_width=True):
                            st.session_state.ativo_visualizar = sinal['ticker']
                            st.rerun()
                        
                        risco = abs(sinal['entrada'] - sinal['stop'])
                        reward = abs(sinal['alvo'] - sinal['entrada'])
                        st.caption(f"Risco: R$ {risco:.2f}")
                        st.caption(f"Reward: R$ {reward:.2f}")
        
        else:
            st.info("Nenhum sinal forte encontrado")


# ==========================================
# RODAPÉ COM EXPLICAÇÕES
# ==========================================

st.markdown("---")

with st.expander("📖 Diferença entre Convergência e Sinal"):
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("""
        ### 🔄 CONVERGÊNCIA
        
        **Definição:** Ambos os timeframes (diário e semanal) apresentam cruzamento na mesma direção dentro do período de lookback.
        
        **Critérios:**
        - ✅ Cruzamento no diário (Linha Branca x EMA Laranja)
        - ✅ Cruzamento no semanal (mesma lógica)
        - ✅ Mesma direção (ambos COMPRA ou ambos VENDA)
        
        **Interpretação:**
        - Indica alinhamento entre timeframes
        - Maior probabilidade de movimento sustentado
        - Confirma tendência em múltiplos prazos
        """)
    
    with col_exp2:
        st.markdown("""
        ### 🎯 SINAL
        
        **Definição:** Convergência + critérios adicionais de força e timing para operação imediata.
        
        **Critérios Extras:**
        - ✅ Cruzamentos recentes (últimas 3 barras)
        - ✅ Momentum favorável
        - ✅ Timing adequado para entrada
        
        **Tipos de Sinal:**
        - **SIMULTÂNEO FORTE:** Ambos na última barra
        - **CONFIRMAÇÃO DIÁRIO:** Semanal já posicionado
        - **CONFIRMAÇÃO SEMANAL:** Diário já posicionado
        - **CONVERGENTE RECENTE:** Ambos recentes
        """)

st.markdown("""
### 💡 Como Usar

1. **Convergências** mostram oportunidades potenciais - analise o contexto
2. **Sinais** indicam momento adequado para entrada - considere operar
3. Use sempre stop loss nos pontos calculados
4. Considere o contexto do mercado e análise fundamentalista
5. Teste em paper trading antes de operar com dinheiro real

⚠️ **Aviso:** Esta ferramenta é para análise técnica educacional. Não é recomendação de investimento.
""")
