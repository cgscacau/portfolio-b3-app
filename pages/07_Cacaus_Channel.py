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
# OBTENÇÃO DE DADOS OHLC REAIS
# ==========================================

def obter_dados_ohlc_reais(ticker, data_inicio, data_fim):
    """
    Obtém dados OHLC reais - primeiro tenta core.data, depois yfinance
    """
    try:
        # Tentar primeiro com core.data
        df_original = get_price_history([ticker], data_inicio, data_fim)
        
        # Verificar se retornou dados OHLC completos
        if not df_original.empty:
            # Se é MultiIndex com OHLC
            if isinstance(df_original.columns, pd.MultiIndex):
                if ticker in df_original.columns.get_level_values(0):
                    df_ticker = df_original[ticker]
                    if all(col in df_ticker.columns for col in ['Open', 'High', 'Low', 'Close']):
                        ohlc = df_ticker[['Open', 'High', 'Low', 'Close']].copy()
                        ohlc['Volume'] = df_ticker.get('Volume', 0)
                        return ohlc.dropna()
            
            # Se já tem colunas OHLC diretas
            elif all(col in df_original.columns for col in ['Open', 'High', 'Low', 'Close']):
                ohlc = df_original[['Open', 'High', 'Low', 'Close']].copy()
                ohlc['Volume'] = df_original.get('Volume', 0)
                return ohlc.dropna()
    except:
        pass
    
    # Fallback para yfinance
    try:
        import yfinance as yf
        
        ticker_yf = ticker + ".SA" if not ticker.endswith(".SA") else ticker
        
        # Adicionar margem para garantir dados suficientes
        data_inicio_buffer = data_inicio - timedelta(days=60)
        
        stock = yf.Ticker(ticker_yf)
        df_yf = stock.history(
            start=data_inicio_buffer, 
            end=data_fim + timedelta(days=1), 
            auto_adjust=True
        )
        
        if not df_yf.empty:
            # Filtrar para período solicitado
            df_yf = df_yf[df_yf.index >= data_inicio]
            return df_yf[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    except ImportError:
        st.error("⚠️ Para dados OHLC reais, instale: pip install yfinance")
    except Exception as e:
        st.warning(f"Erro ao obter dados de {ticker}: {str(e)}")
    
    return pd.DataFrame()


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
        return ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]  # Fallback


# ==========================================
# CÁLCULOS DO INDICADOR
# ==========================================

def calcular_cacaus_channel(df, periodo_superior=20, periodo_inferior=30, ema_periodo=9):
    """Calcula o indicador Cacau's Channel"""
    if df.empty or len(df) < max(periodo_superior, periodo_inferior, ema_periodo):
        return df
    
    df = df.copy()
    df['linha_superior'] = df['High'].rolling(window=periodo_superior).max()
    df['linha_inferior'] = df['Low'].rolling(window=periodo_inferior).min()
    df['linha_media'] = (df['linha_superior'] + df['linha_inferior']) / 2
    df['ema_media'] = df['linha_media'].ewm(span=ema_periodo, adjust=False).mean()
    
    return df


def resample_para_semanal(df):
    """Converte dados diários para semanais"""
    if df.empty or len(df) < 5:
        return pd.DataFrame()
    
    return df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()


def detectar_cruzamentos_e_convergencia(df_diario, df_semanal, lookback=5):
    """
    Detecta cruzamentos e diferencia CONVERGÊNCIA de SINAL
    
    CONVERGÊNCIA: Cruzamentos na mesma direção em ambos timeframes
    SINAL: Convergência + pelo menos um cruzamento na última barra
    """
    
    def encontrar_cruzamento(df):
        """Encontra o cruzamento mais recente dentro do lookback"""
        for i in range(1, min(lookback + 1, len(df))):
            atual = df.iloc[-i]
            anterior = df.iloc[-(i+1)]
            
            if pd.isna(atual['linha_media']) or pd.isna(atual['ema_media']):
                continue
            
            # Cruzamento para CIMA
            if (anterior['linha_media'] <= anterior['ema_media'] and 
                atual['linha_media'] > atual['ema_media']):
                return 'COMPRA', i
            
            # Cruzamento para BAIXO
            if (anterior['linha_media'] >= anterior['ema_media'] and 
                atual['linha_media'] < atual['ema_media']):
                return 'VENDA', i
        
        return None, None
    
    # Detectar cruzamentos
    cruz_diario, barras_diario = encontrar_cruzamento(df_diario.dropna())
    cruz_semanal, barras_semanal = encontrar_cruzamento(df_semanal.dropna())
    
    # Verificar CONVERGÊNCIA
    tem_convergencia = (cruz_diario and cruz_semanal and cruz_diario == cruz_semanal)
    
    # Verificar SINAL (convergência + gatilho na última barra)
    tem_sinal = tem_convergencia and (barras_diario == 1 or barras_semanal == 1)
    
    # Classificar tipo
    tipo = None
    if tem_convergencia:
        if barras_diario == 1 and barras_semanal == 1:
            tipo = 'SIMULTÂNEO'
        elif barras_diario == 1:
            tipo = 'GATILHO DIÁRIO'
        elif barras_semanal == 1:
            tipo = 'GATILHO SEMANAL'
        else:
            tipo = 'CONVERGÊNCIA RECENTE'
    
    return {
        'tem_convergencia': tem_convergencia,
        'tem_sinal': tem_sinal,
        'direcao': cruz_diario if tem_convergencia else None,
        'tipo': tipo,
        'cruz_diario': cruz_diario,
        'cruz_semanal': cruz_semanal,
        'barras_diario': barras_diario,
        'barras_semanal': barras_semanal
    }


def calcular_pontos_operacao(df, direcao, rr_ratio=2.0):
    """Calcula entrada, stop e alvo"""
    if df.empty:
        return {'entrada': 0, 'stop': 0, 'alvo': 0}
    
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
    
    return {'entrada': entrada, 'stop': stop, 'alvo': alvo}


# ==========================================
# VISUALIZAÇÃO - GRÁFICOS DUPLOS
# ==========================================

def criar_graficos_duplos(df_diario, df_semanal, ticker):
    """Cria gráficos lado a lado com velas OHLC completas"""
    
    # Limitar dados para visualização
    df_d = df_diario.tail(100).dropna()
    df_s = df_semanal.tail(50).dropna()
    
    if df_d.empty or df_s.empty:
        return None
    
    # Criar subplots lado a lado
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{ticker} - Diário', f'{ticker} - Semanal'),
        horizontal_spacing=0.08
    )
    
    # GRÁFICO DIÁRIO
    fig.add_trace(
        go.Candlestick(
            x=df_d.index,
            open=df_d['Open'],
            high=df_d['High'],
            low=df_d['Low'],
            close=df_d['Close'],
            name='Preço',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Indicadores diário
    for nome, coluna, cor, largura in [
        ('Superior', 'linha_superior', '#ff4444', 2),
        ('Inferior', 'linha_inferior', '#00ff00', 2),
        ('Média', 'linha_media', 'white', 2.5),
        ('EMA', 'ema_media', '#ff9800', 2)
    ]:
        fig.add_trace(
            go.Scatter(
                x=df_d.index, 
                y=df_d[coluna],
                mode='lines',
                name=nome,
                line=dict(
                    color=cor, 
                    width=largura,
                    dash='dash' if nome == 'EMA' else 'solid'
                ),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # GRÁFICO SEMANAL
    fig.add_trace(
        go.Candlestick(
            x=df_s.index,
            open=df_s['Open'],
            high=df_s['High'],
            low=df_s['Low'],
            close=df_s['Close'],
            name='Preço',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Indicadores semanal
    for nome, coluna, cor, largura in [
        ('Superior', 'linha_superior', '#ff4444', 2),
        ('Inferior', 'linha_inferior', '#00ff00', 2),
        ('Média', 'linha_media', 'white', 2.5),
        ('EMA', 'ema_media', '#ff9800', 2)
    ]:
        fig.add_trace(
            go.Scatter(
                x=df_s.index, 
                y=df_s[coluna],
                mode='lines',
                name=nome,
                line=dict(
                    color=cor, 
                    width=largura,
                    dash='dash' if nome == 'EMA' else 'solid'
                ),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Layout otimizado
    fig.update_layout(
        height=600,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        title={
            'text': f"{ticker} - Cacau's Channel (Diário | Semanal)",
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    # Remover rangesliders
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

st.set_page_config(layout="wide", page_title="Cacau's Channel Screener")

st.title("🎯 Cacau's Channel - Screener Multi-Timeframe")
st.markdown("**CONVERGÊNCIA** = Cruzamentos alinhados | **SINAL** = Convergência + Gatilho na última barra")

try:
    cache_manager.exibir_painel_controle()
except:
    pass

st.markdown("---")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ Configurações")
    
    periodo_superior = st.number_input("Período Superior", 5, 50, 20)
    periodo_inferior = st.number_input("Período Inferior", 5, 50, 30)
    ema_periodo = st.number_input("EMA Período", 3, 30, 9)
    rr_ratio = st.selectbox("Risk/Reward", [1.5, 2.0, 2.5, 3.0], index=1, format_func=lambda x: f"1:{x}")
    lookback = st.number_input("Lookback Cruzamento", 1, 10, 5)
    
    st.markdown("---")
    
    data_fim = st.date_input("Data Final", datetime.now())
    periodo_str = st.selectbox("Período", ["3 meses", "6 meses", "1 ano", "2 anos", "3 anos", "5 anos"], index=2)
    
    periodos = {"3 meses": 90, "6 meses": 180, "1 ano": 365, "2 anos": 730, "3 anos": 1095, "5 anos": 1825}
    data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=periodos[periodo_str])
    data_fim_dt = datetime.combine(data_fim, datetime.min.time())

# ==========================================
# LAYOUT PRINCIPAL
# ==========================================

col_config, col_graficos = st.columns([1, 2])

with col_config:
    st.subheader("📈 Ativos")
    
    base_ativos = carregar_base_ativos()
    
    opcao = st.radio("Fonte", ["📁 Portfólio", "🌐 Base B3", "✍️ Manual"], index=2)
    
    tickers = []
    
    if opcao == "📁 Portfólio":
        try:
            from core.portfolio import listar_portfolios, carregar_portfolio
            portfolios = listar_portfolios()
            if portfolios:
                port_sel = st.selectbox("Portfólio", portfolios)
                portfolio = carregar_portfolio(port_sel)
                tickers = portfolio.tickers if portfolio else []
        except:
            st.warning("Módulo portfolio não encontrado")
    
    elif opcao == "🌐 Base B3":
        filtros = st.multiselect("Tipos", ["Ações", "FIIs", "ETFs"], default=["Ações"])
        limite = st.number_input("Limite", 10, 100, 30)
        
        if "Ações" in filtros:
            tickers.extend([t for t in base_ativos if t[-1] in ['3','4'] and not t.endswith('11')])
        if "FIIs" in filtros:
            tickers.extend([t for t in base_ativos if t.endswith('11')])
        if "ETFs" in filtros:
            tickers.extend([t for t in base_ativos if 'B' in t[-2:]])
        
        tickers = sorted(list(set(tickers)))[:limite]
    
    else:  # Manual
        entrada = st.text_area("Tickers", "ALPA4\nPETR4\nVALE3\nITUB4", height=100)
        tickers = [t.strip().upper() for t in entrada.split('\n') if t.strip()]
    
    st.caption(f"📊 {len(tickers)} ativo(s)")
    
    # SCREENER
    if st.button("🔍 Executar Screener", type="primary", use_container_width=True):
        convergencias = []
        sinais = []
        todos_dados = {}
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, ticker in enumerate(tickers):
            progress.progress((i+1)/len(tickers))
            status.text(f"Analisando {ticker}...")
            
            try:
                df_ohlc = obter_dados_ohlc_reais(ticker, data_inicio, data_fim_dt)
                
                if df_ohlc.empty or len(df_ohlc) < 60:
                    continue
                
                df_diario = calcular_cacaus_channel(df_ohlc, periodo_superior, periodo_inferior, ema_periodo)
                df_semanal_raw = resample_para_semanal(df_ohlc)
                
                if df_semanal_raw.empty:
                    continue
                
                df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                
                resultado = detectar_cruzamentos_e_convergencia(df_diario, df_semanal, lookback)
                
                todos_dados[ticker] = {
                    'df_diario': df_diario,
                    'df_semanal': df_semanal,
                    'resultado': resultado
                }
                
                if resultado['tem_convergencia']:
                    pontos = calcular_pontos_operacao(df_diario, resultado['direcao'], rr_ratio)
                    
                    item = {
                        'ticker': ticker,
                        'direcao': resultado['direcao'],
                        'tipo': resultado['tipo'],
                        'entrada': pontos['entrada'],
                        'stop': pontos['stop'],
                        'alvo': pontos['alvo']
                    }
                    
                    convergencias.append(item)
                    
                    if resultado['tem_sinal']:
                        sinais.append(item)
            
            except Exception as e:
                continue
        
        progress.empty()
        status.empty()
        
        st.session_state.convergencias = convergencias
        st.session_state.sinais = sinais
        st.session_state.todos_dados = todos_dados
        
        st.metric("🔄 Convergências", len(convergencias))
        st.metric("🎯 Sinais", len(sinais))
    
    # LISTAS DE RESULTADOS
    st.markdown("---")
    
    # SINAIS (prioritários)
    st.subheader("🎯 Sinais (Gatilho Ativo)")
    if 'sinais' in st.session_state and st.session_state.sinais:
        for sinal in st.session_state.sinais:
            cor = "🟢" if sinal['direcao'] == 'COMPRA' else "🔴"
            if st.button(f"{cor} {sinal['ticker']} - {sinal['tipo']}", 
                        key=f"sinal_{sinal['ticker']}", use_container_width=True):
                st.session_state.ativo_selecionado = sinal['ticker']
                st.rerun()
    else:
        st.caption("Nenhum sinal ativo encontrado")
    
    # CONVERGÊNCIAS (informativas)
    st.subheader("🔄 Convergências Gerais")
    if 'convergencias' in st.session_state and st.session_state.convergencias:
        # Mostrar apenas convergências que NÃO são sinais
        sinais_tickers = {s['ticker'] for s in st.session_state.get('sinais', [])}
        conv_apenas = [c for c in st.session_state.convergencias if c['ticker'] not in sinais_tickers]
        
        for conv in conv_apenas:
            cor = "🟢" if conv['direcao'] == 'COMPRA' else "🔴"
            if st.button(f"{cor} {conv['ticker']} - {conv['tipo']}", 
                        key=f"conv_{conv['ticker']}", use_container_width=True):
                st.session_state.ativo_selecionado = conv['ticker']
                st.rerun()
    else:
        st.caption("Execute o screener")

# ==========================================
# COLUNA GRÁFICOS
# ==========================================

with col_graficos:
    if 'todos_dados' in st.session_state and st.session_state.todos_dados:
        
        ativos_disponiveis = sorted(st.session_state.todos_dados.keys())
        ativo_default = st.session_state.get('ativo_selecionado', ativos_disponiveis[0])
        
        ativo = st.selectbox("Ativo", ativos_disponiveis, 
                           index=ativos_disponiveis.index(ativo_default) if ativo_default in ativos_disponiveis else 0)
        
        dados = st.session_state.todos_dados[ativo]
        resultado = dados['resultado']
        
        # STATUS
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if resultado['tem_sinal']:
                st.success(f"🎯 SINAL: {resultado['direcao']}")
            elif resultado['tem_convergencia']:
                st.info(f"🔄 CONVERGÊNCIA: {resultado['direcao']}")
            else:
                st.warning("❌ Sem convergência")
        
        with col2:
            if resultado['cruz_diario']:
                cor = "🟢" if resultado['cruz_diario'] == 'COMPRA' else "🔴"
                st.write(f"📅 Diário: {cor} {resultado['cruz_diario']} ({resultado['barras_diario']})")
        
        with col3:
            if resultado['cruz_semanal']:
                cor = "🟢" if resultado['cruz_semanal'] == 'COMPRA' else "🔴"
                st.write(f"📆 Semanal: {cor} {resultado['cruz_semanal']} ({resultado['barras_semanal']})")
        
        # PONTOS DE OPERAÇÃO
        if resultado['tem_convergencia']:
            pontos = calcular_pontos_operacao(dados['df_diario'], resultado['direcao'], rr_ratio)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Entrada", f"R$ {pontos['entrada']:.2f}")
            col2.metric("Stop", f"R$ {pontos['stop']:.2f}")
            col3.metric("Alvo", f"R$ {pontos['alvo']:.2f}")
        
        # GRÁFICOS DUPLOS
        fig = criar_graficos_duplos(dados['df_diario'], dados['df_semanal'], ativo)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Erro ao gerar gráficos")
    
    else:
        st.info("👈 Execute o screener para visualizar")

# ==========================================
# EXPLICAÇÃO
# ==========================================

st.markdown("---")

with st.expander("📖 Diferença: Convergência vs Sinal"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔄 CONVERGÊNCIA
        
        **Definição:** Cruzamentos na mesma direção em ambos timeframes (dentro do lookback)
        
        **Critérios:**
        - ✅ Cruzamento Diário: Linha Branca × EMA Laranja
        - ✅ Cruzamento Semanal: Mesma lógica
        - ✅ Mesma direção (COMPRA ou VENDA)
        
        **Tipos:**
        - **SIMULTÂNEO:** Ambos na última barra
        - **GATILHO DIÁRIO:** Semanal posicionado + Diário cruzou
        - **GATILHO SEMANAL:** Diário posicionado + Semanal cruzou
        - **CONVERGÊNCIA RECENTE:** Ambos recentes, sem gatilho
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 SINAL
        
        **Definição:** Convergência + gatilho na última barra
        
        **Critérios Extras:**
        - ✅ Convergência confirmada
        - ✅ Pelo menos um cruzamento na última barra
        - ✅ Momento adequado para entrada
        
        **Diferença:**
        - Convergência = **Alinhamento** entre timeframes
        - Sinal = **Momento de ação** (entrada recomendada)
        
        **Uso:**
        - **Sinais:** Considere para operação imediata
        - **Convergências:** Monitore para possível sinal futuro
        """)

st.markdown("""
### 💡 Como Interpretar

- **🎯 Sinais:** Oportunidades com gatilho ativo - considere entrada
- **🔄 Convergências:** Situações alinhadas mas sem urgência - monitore  
- **Gráficos Duplos:** Compare diário (timing) vs semanal (tendência)
- **Stop/Alvo:** Sempre use gestão de risco baseada no canal

⚠️ **Aviso:** Ferramenta educacional. Não é recomendação de investimento.
""")
