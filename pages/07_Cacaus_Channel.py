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

# Importar módulos (assumindo que core.data e core.cache existem e funcionam)
try:
    from core.data import get_price_history
    from core.cache import cache_manager
except ImportError:
    st.warning("Módulos 'core.data' ou 'core.cache' não encontrados. Usando mock functions.")
    
    # Mock functions para que o código possa rodar sem os módulos externos
    class MockCacheManager:
        def exibir_painel_controle(self):
            st.info("Painel de controle do cache (mock) desabilitado.")
    cache_manager = MockCacheManager()

    def get_price_history(tickers, start_date, end_date):
        """Mock function para get_price_history."""
        st.warning(f"Usando mock para get_price_history. Dados para {tickers} de {start_date} a {end_date} serão gerados artificialmente.")
        if not tickers:
            return pd.DataFrame()
        
        # Gerar dados OHLCV artificiais
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        if len(dates) == 0:
            return pd.DataFrame()

        data = []
        for ticker in tickers:
            # Gerar um preço base aleatório
            base_price = np.random.uniform(5, 100)
            prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.5)
            prices = np.maximum(prices, 1.0) # Ensure prices are not negative

            df_ticker = pd.DataFrame({
                'Open': prices,
                'High': prices * (1 + np.random.uniform(0.001, 0.01, len(dates))),
                'Low': prices * (1 - np.random.uniform(0.001, 0.01, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(100000, 10000000, len(dates))
            }, index=dates)
            df_ticker.columns = pd.MultiIndex.from_product([[ticker], df_ticker.columns])
            data.append(df_ticker)
        
        if data:
            return pd.concat(data, axis=1)
        return pd.DataFrame()


# ==========================================
# CARREGAR BASE DE ATIVOS
# ==========================================

@st.cache_data
def carregar_base_ativos():
    """Carrega base completa de ativos da B3"""
    try:
        # Assumindo que 'assets/b3_universe.csv' existe
        # Se não existir, pode ser necessário criar um arquivo mock ou ajustar o caminho
        caminho = os.path.join('assets', 'b3_universe.csv')
        
        # Criar um arquivo mock se não existir para fins de demonstração
        if not os.path.exists('assets'):
            os.makedirs('assets')
        if not os.path.exists(caminho):
            with open(caminho, 'w') as f:
                f.write("ticker\nPETR4\nVALE3\nITUB4\nBBDC4\nABEV3\nALPA4\nB3SA3\nWEGE3\nFLRY3")
            st.warning("Arquivo 'b3_universe.csv' não encontrado, um arquivo mock foi criado.")

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
        # Retorna alguns tickers de exemplo se houver erro
        return ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "ALPA4"]


# ==========================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# ==========================================

def criar_ohlc_correto(df_precos_raw, ticker):
    """
    Cria DataFrame OHLC correto a partir dos dados de preços históricos.
    Prioriza colunas OHLCV se presentes. Se apenas o ticker (Close) estiver presente,
    gera OHLC aproximado.
    """
    df_ohlc = pd.DataFrame(index=df_precos_raw.index)

    # Verifica se o df_precos_raw é um MultiIndex DataFrame (output comum de get_price_history para múltiplos tickers)
    if isinstance(df_precos_raw.columns, pd.MultiIndex):
        if ticker in df_precos_raw.columns.levels[0]:
            df_ticker_data = df_precos_raw[ticker]
            # Verifica se as colunas OHLCV estão presentes para o ticker específico
            if all(col in df_ticker_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                df_ohlc = df_ticker_data[['Open', 'High', 'Low', 'Close']].copy()
                df_ohlc['Volume'] = df_ticker_data.get('Volume', 0)
            elif 'Close' in df_ticker_data.columns:
                # Se apenas Close está disponível para o ticker
                precos_fechamento = df_ticker_data['Close'].dropna()
                df_ohlc['Close'] = precos_fechamento
                df_ohlc['Open'] = precos_fechamento.shift(1).fillna(precos_fechamento)
                df_ohlc['High'] = df_ohlc[['Open', 'Close']].max(axis=1) * 1.005
                df_ohlc['Low'] = df_ohlc[['Open', 'Close']].min(axis=1) * 0.995
                df_ohlc['Volume'] = 0
            else:
                return pd.DataFrame() # Nenhum dado relevante encontrado para o ticker
        else:
            return pd.DataFrame() # Ticker não encontrado no MultiIndex
    elif all(col in df_precos_raw.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Se o DataFrame já tem as colunas OHLCV no nível superior (para um único ticker)
        df_ohlc = df_precos_raw[['Open', 'High', 'Low', 'Close']].copy()
        df_ohlc['Volume'] = df_precos_raw.get('Volume', 0)
    elif ticker in df_precos_raw.columns:
        # Se o DataFrame tem apenas uma coluna com o nome do ticker (assumido como Close)
        precos_fechamento = df_precos_raw[ticker].dropna()
        df_ohlc['Close'] = precos_fechamento
        df_ohlc['Open'] = precos_fechamento.shift(1).fillna(precos_fechamento)
        df_ohlc['High'] = df_ohlc[['Open', 'Close']].max(axis=1) * 1.005
        df_ohlc['Low'] = df_ohlc[['Open', 'Close']].min(axis=1) * 0.995
        df_ohlc['Volume'] = 0
    else:
        return pd.DataFrame() # Nenhum formato de dados conhecido

    return df_ohlc.dropna()


def obter_dados_historicos_completos(ticker, data_inicio, data_fim, max_tentativas=3):
    """
    Obtém dados históricos com múltiplas tentativas e validação.
    Ajusta a data de início para garantir dados suficientes para os indicadores.
    """
    # Adicionar uma margem de segurança para garantir dados para os cálculos de rolling window e EMA
    # Por exemplo, 200 dias extras para cobrir períodos de 5 anos + indicadores
    data_inicio_real = data_inicio - timedelta(days=200) 
    
    for tentativa in range(max_tentativas):
        try:
            df = get_price_history([ticker], data_inicio_real, data_fim)
            
            if df.empty:
                continue
            
            df_ohlc = criar_ohlc_correto(df, ticker)
            
            if df_ohlc.empty:
                continue

            # Filtrar para o período solicitado pelo usuário após os cálculos do indicador
            df_ohlc = df_ohlc[df_ohlc.index >= data_inicio]
            
            # Garantir que temos dados suficientes após o filtro
            if len(df_ohlc) >= 50: # Mínimo de 50 barras para ser útil
                return df_ohlc
            
        except Exception as e:
            if tentativa == max_tentativas - 1:
                st.warning(f"Erro ao obter dados de {ticker} após {max_tentativas} tentativas: {str(e)}")
            continue
    
    return pd.DataFrame()


# ==========================================
# FUNÇÕES DE CÁLCULO DO INDICADOR
# ==========================================

def calcular_cacaus_channel(df, periodo_superior=20, periodo_inferior=30, ema_periodo=9):
    """Calcula o indicador Cacau's Channel"""
    df = df.copy()
    
    # Garantir que temos dados suficientes para os cálculos
    min_period = max(periodo_superior, periodo_inferior, ema_periodo)
    if len(df) < min_period + 1: # +1 para permitir shift ou cálculo da primeira EMA
        return df # Retorna DF original, talvez vazio ou com poucos dados
    
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
    if df.empty or len(df) < 5: # Mínimo de 5 dias para formar 1 semana
        return pd.DataFrame()
    
    try:
        df_semanal = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Ajustar o índice para o início da semana para consistência, se desejado
        # df_semanal.index = df_semanal.index - pd.Timedelta(days=6) # Ajusta para segunda-feira
        
        return df_semanal
    except Exception as e:
        st.error(f"Erro ao criar timeframe semanal: {str(e)}")
        return pd.DataFrame()


def detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback=5):
    """
    Detecta convergência de CRUZAMENTOS entre timeframes.
    Retorna o status dos cruzamentos e se há convergência.
    """
    
    # Garantir que há dados suficientes para o lookback
    if len(df_diario.dropna()) < lookback + 1 or len(df_semanal.dropna()) < lookback + 1:
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
    df_diario_clean = df_diario.dropna(subset=['linha_media', 'ema_media'])
    
    for i in range(1, min(lookback + 1, len(df_diario_clean))):
        idx_atual = -i
        idx_anterior = -(i+1)
        
        linha_media_atual = df_diario_clean['linha_media'].iloc[idx_atual]
        ema_media_atual = df_diario_clean['ema_media'].iloc[idx_atual]
        linha_media_anterior = df_diario_clean['linha_media'].iloc[idx_anterior]
        ema_media_anterior = df_diario_clean['ema_media'].iloc[idx_anterior]
        
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
    df_semanal_clean = df_semanal.dropna(subset=['linha_media', 'ema_media'])
    
    for i in range(1, min(lookback + 1, len(df_semanal_clean))):
        idx_atual = -i
        idx_anterior = -(i+1)
        
        linha_media_atual = df_semanal_clean['linha_media'].iloc[idx_atual]
        ema_media_atual = df_semanal_clean['ema_media'].iloc[idx_atual]
        linha_media_anterior = df_semanal_clean['linha_media'].iloc[idx_anterior]
        ema_media_anterior = df_semanal_clean['ema_media'].iloc[idx_anterior]
        
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
                tipo_sinal = 'RECENTE' # Convergência, mas não na última barra
    
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
    if df.empty:
        return {'entrada': np.nan, 'stop': np.nan, 'alvo': np.nan, 'rr': f"1:{rr_ratio}"}

    ultima_linha = df.iloc[-1]
    entrada = ultima_linha['Close']
    
    if direcao == 'COMPRA':
        stop = ultima_linha['linha_inferior']
        distancia = entrada - stop
        alvo = entrada + (distancia * rr_ratio)
    else: # VENDA
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

def criar_grafico_cacaus_channel(df_diario, df_semanal, ticker, timeframe_ativo="Diário", num_barras=100):
    """Cria gráfico do Cacau's Channel com candlesticks completos e bem centralizado"""
    
    df = df_diario if timeframe_ativo == "Diário" else df_semanal
    
    # Determinar número de barras a mostrar
    df = df.tail(num_barras).copy()
    
    # Verificar se temos dados suficientes
    if df.empty or len(df) < 5: # Mínimo de 5 barras para um gráfico significativo
        # st.warning(f"Dados insuficientes para gerar gráfico {timeframe_ativo} de {ticker}")
        return None
    
    fig = go.Figure()
    
    # Candlestick com OHLC correto
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Preço',
        increasing_line_color='#26a69a', # Verde para alta
        decreasing_line_color='#ef5350', # Vermelho para baixa
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        whiskerwidth=0.5,
        increasing_line_width=1.5,
        decreasing_line_width=1.5
    ))
    
    # Linha Superior (vermelha)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_superior'],
        mode='lines',
        name='Linha Superior',
        line=dict(color='#ff4444', width=2), # Vermelho vibrante
        hovertemplate='Superior: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Linha Inferior (verde)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_inferior'],
        mode='lines',
        name='Linha Inferior',
        line=dict(color='#00ff00', width=2), # Verde vibrante
        hovertemplate='Inferior: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Linha Média (branca)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_media'],
        mode='lines',
        name='Linha Média',
        line=dict(color='white', width=2.5),
        hovertemplate='Média: R$ %{y:.2f}<extra></extra>'
    ))
    
    # EMA da Média (laranja)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema_media'],
        mode='lines',
        name='EMA Média',
        line=dict(color='#ff9800', width=2.5, dash='dash'), # Laranja vibrante
        hovertemplate='EMA: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Calcular range de preços para melhor centralização
    # Incluir as linhas do indicador no cálculo do range
    all_prices = pd.concat([df['Low'], df['High'], df['linha_superior'], df['linha_inferior'], df['linha_media'], df['ema_media']]).dropna()
    
    if not all_prices.empty:
        preco_min = all_prices.min()
        preco_max = all_prices.max()
        margem = (preco_max - preco_min) * 0.1  # 10% de margem
        y_axis_range = [preco_min - margem, preco_max + margem]
    else:
        y_axis_range = [df['Close'].min() * 0.9, df['Close'].max() * 1.1] # Fallback
    
    fig.update_layout(
        title={
            'text': f"{ticker} - Cacau's Channel ({timeframe_ativo})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        height=500, # Altura ajustada para caber dois gráficos
        template="plotly_dark",
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        
        # Melhorar centralização e margens
        margin=dict(l=60, r=60, t=80, b=60), # Margens ajustadas
        
        # Configurar eixo Y para melhor visualização
        yaxis=dict(
            range=y_axis_range,
            autorange=False,
            fixedrange=False,
            showgrid=True, gridwidth=0.5, gridcolor='#333333'
        ),
        
        # Configurar eixo X
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date',
            showgrid=True, gridwidth=0.5, gridcolor='#333333'
        ),
        
        # Legenda
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0.5)' # Fundo semi-transparente para legenda
        ),
        
        # Cor de fundo
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    return fig


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

st.set_page_config(layout="wide", page_title="Cacau's Channel Screener")

st.title("🎯 Cacau's Channel - Screener")
st.markdown("Screener automático com detecção de cruzamentos e convergência")

try:
    cache_manager.exibir_painel_controle()
except:
    pass

st.markdown("---")


# Inicializar session_state para evitar KeyErrors
if 'cacaus_sinais_acionaveis' not in st.session_state:
    st.session_state.cacaus_sinais_acionaveis = []
if 'cacaus_convergencias_gerais' not in st.session_state:
    st.session_state.cacaus_convergencias_gerais = []
if 'cacaus_todos_dados' not in st.session_state:
    st.session_state.cacaus_todos_dados = {}
if 'ativo_visualizar' not in st.session_state:
    st.session_state.ativo_visualizar = None


# ==========================================
# SIDEBAR - CONFIGURAÇÕES
# ==========================================

with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.subheader("📊 Parâmetros do Indicador")
    
    periodo_superior = st.number_input("Período Linha Superior (High)", min_value=5, max_value=50, value=20, step=1)
    periodo_inferior = st.number_input("Período Linha Inferior (Low)", min_value=5, max_value=50, value=30, step=1)
    ema_periodo = st.number_input("EMA Período (Média)", min_value=3, max_value=30, value=9, step=1)
    rr_ratio = st.selectbox("Risk/Reward (1:X)", options=[1.5, 2.0, 2.5, 3.0], index=1, format_func=lambda x: f"1:{x}")
    lookback_cruzamento = st.number_input("Lookback Cruzamento", min_value=1, max_value=10, value=5, step=1, 
                                          help="Quantas barras olhar para trás para detectar cruzamento")
    num_barras_grafico = st.number_input("Barras no Gráfico", min_value=20, max_value=200, value=100, step=10, 
                                         help="Número de barras a exibir nos gráficos de candlestick.")
    
    st.markdown("---")
    
    st.subheader("📅 Período de Análise")
    
    data_fim = st.date_input("Data Final", value=datetime.now(), max_value=datetime.now())
    
    periodo_analise_str = st.selectbox(
        "Duração do Período",
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
    
    dias_periodo = periodos_dias[periodo_analise_str]
    data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=dias_periodo)
    data_fim_dt = datetime.combine(data_fim, datetime.min.time())
    
    st.info(f"📊 Analisando de {data_inicio.strftime('%d/%m/%Y')} até {data_fim.strftime('%d/%m/%Y')}")


# ==========================================
# LAYOUT EM DUAS COLUNAS
# ==========================================

col_esquerda, col_direita = st.columns([1, 3])


# ==========================================
# COLUNA ESQUERDA: SELEÇÃO E SCREENER
# ==========================================

with col_esquerda:
    
    st.subheader("📈 Seleção de Ativos")
    
    # Carregar base completa
    base_completa = carregar_base_ativos()
    
    if base_completa:
        st.caption(f"✅ {len(base_completa)} ativos disponíveis na base.")
    
    # Opções de seleção
    opcao_selecao = st.radio(
        "Fonte de Ativos",
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
                portfolio_selecionado = st.selectbox("Selecione o portfólio", portfolios_disponiveis, label_visibility="collapsed")
                portfolio = carregar_portfolio(portfolio_selecionado)
                tickers = portfolio.tickers if portfolio else []
                st.caption(f"📊 {len(tickers)} ativos no portfólio selecionado.")
            else:
                st.warning("Nenhum portfólio encontrado. Crie um ou use outra fonte.")
        except ImportError:
            st.warning("Módulo 'core.portfolio' não encontrado. Selecione outra fonte de ativos.")
        except Exception as e:
            st.error(f"Erro ao carregar portfólios: {str(e)}")
    
    # OPÇÃO 2: Base B3
    elif opcao_selecao == "🌐 Base B3":
        if base_completa:
            
            filtro_tipo = st.multiselect(
                "Filtrar por Tipo de Ativo",
                options=["Ações", "FIIs", "ETFs", "Todos"],
                default=["Ações"],
                label_visibility="collapsed"
            )
            
            limite_ativos = st.number_input(
                "Limite de Ativos para Análise",
                min_value=10,
                max_value=min(500, len(base_completa)), # Limite razoável para evitar sobrecarga
                value=50,
                step=10,
                label_visibility="collapsed",
                help="Número máximo de ativos a serem processados pelo screener."
            )
            
            if "Todos" in filtro_tipo:
                tickers = base_completa
            else:
                tickers_filtrados = []
                
                if "Ações" in filtro_tipo:
                    tickers_filtrados.extend([t for t in base_completa if (t.endswith('3') or t.endswith('4')) and not t.endswith('11')])
                
                if "FIIs" in filtro_tipo:
                    tickers_filtrados.extend([t for t in base_completa if t.endswith('11')])
                
                if "ETFs" in filtro_tipo:
                    # ETFs geralmente terminam com 'B' e um número ou apenas 'B'
                    tickers_filtrados.extend([t for t in base_completa if 'B' in t[-2:] and not (t.endswith('3') or t.endswith('4') or t.endswith('11'))])
                
                tickers = sorted(list(set(tickers_filtrados)))
            
            if limite_ativos > 0 and len(tickers) > limite_ativos:
                tickers = tickers[:limite_ativos]
            
            st.caption(f"📊 {len(tickers)} ativos selecionados para o screener.")
        else:
            st.warning("Base de dados B3 não carregada. Verifique o arquivo 'b3_universe.csv'.")
    
    # OPÇÃO 3: Manual
    elif opcao_selecao == "✍️ Manual":
        tickers_input = st.text_area(
            "Digite os tickers (um por linha ou separados por vírgula)",
            value="PETR4\nVALE3\nITUB4",
            height=100,
            label_visibility="collapsed"
        )
        
        tickers_raw = tickers_input.replace(',', '\n').split('\n')
        tickers = [t.strip().upper() for t in tickers_raw if t.strip()]
        
        st.caption(f"📊 {len(tickers)} ativos listados manualmente.")
    
    # Botão de screener
    st.markdown("---")
    
    if st.button("🔍 Executar Screener", type="primary", use_container_width=True):
        
        if not tickers:
            st.error("❌ Nenhum ativo selecionado para o screener. Por favor, escolha os ativos.")
        else:
            sinais_acionaveis = []
            convergencias_gerais = []
            todos_dados = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_analisados = 0
            total_com_dados_suficientes = 0
            total_sinais_acionaveis = 0
            total_convergencias_gerais = 0
            erros = []
            
            for idx, ticker in enumerate(tickers):
                
                progress = (idx + 1) / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"Analisando {ticker}... ({idx+1}/{len(tickers)})")
                
                total_analisados += 1
                
                try:
                    # Obter dados históricos completos
                    df_ativo = obter_dados_historicos_completos(ticker, data_inicio, data_fim_dt)
                    
                    min_bars_needed = max(periodo_superior, periodo_inferior, ema_periodo) + lookback_cruzamento + 5 # Margem
                    if df_ativo.empty or len(df_ativo) < min_bars_needed:
                        erros.append(f"{ticker}: Dados insuficientes para análise. ({len(df_ativo)} barras)")
                        continue
                    
                    total_com_dados_suficientes += 1
                    
                    # Calcular indicador no diário
                    df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
                    
                    # Criar timeframe semanal
                    df_semanal_raw = resample_para_semanal(df_ativo)
                    
                    if df_semanal_raw.empty or len(df_semanal_raw) < min_bars_needed / 5: # Semanal terá menos barras
                        erros.append(f"{ticker}: Erro ao criar dados semanais ou insuficientes.")
                        continue
                    
                    # Calcular indicador no semanal
                    df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                    
                    # Detectar convergência
                    convergencia = detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback_cruzamento)
                    
                    # Salvar TODOS os dados para visualização posterior
                    todos_dados[ticker] = {
                        'df_diario': df_diario,
                        'df_semanal': df_semanal,
                        'convergencia': convergencia
                    }
                    
                    if convergencia['convergente']:
                        total_convergencias_gerais += 1
                        
                        # Adicionar à lista geral de convergências
                        convergencias_gerais.append({
                            'ticker': ticker,
                            'direcao': convergencia['direcao'],
                            'tipo_convergencia': convergencia['tipo_sinal'],
                            'barra_diario': convergencia['barra_cruzamento_diario'],
                            'barra_semanal': convergencia['barra_cruzamento_semanal']
                        })

                        # Adicionar apenas se for um "sinal acionável" (SIMULTÂNEO ou REENTRADA)
                        if convergencia['tipo_sinal'] in ['SIMULTÂNEO', 'REENTRADA DIÁRIO', 'REENTRADA SEMANAL']:
                            total_sinais_acionaveis += 1
                            pontos = calcular_entrada_stop_alvo(df_diario, convergencia['direcao'], rr_ratio)
                            
                            sinais_acionaveis.append({
                                'ticker': ticker,
                                'direcao': convergencia['direcao'],
                                'entrada': pontos['entrada'],
                                'stop': pontos['stop'],
                                'alvo': pontos['alvo'],
                                'rr': pontos['rr'],
                                'tipo_sinal': convergencia['tipo_sinal']
                            })
                
                except Exception as e:
                    erros.append(f"{ticker}: Erro inesperado - {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            # Salvar resultados no session_state
            st.session_state.cacaus_sinais_acionaveis = sinais_acionaveis
            st.session_state.cacaus_convergencias_gerais = convergencias_gerais
            st.session_state.cacaus_todos_dados = todos_dados
            
            # Mostrar estatísticas
            st.markdown("---")
            st.metric("Total Analisados", total_analisados)
            st.metric("Com Dados Suficientes", total_com_dados_suficientes)
            st.metric("Total Convergências", total_convergencias_gerais)
            st.metric("🎯 Sinais Acionáveis", total_sinais_acionaveis)
            
            if sinais_acionaveis:
                st.success(f"✅ {len(sinais_acionaveis)} sinal(is) acionável(is) encontrado(s)!")
            elif convergencias_gerais:
                st.info("ℹ️ Nenhuma sinal acionável, mas há convergências gerais.")
            else:
                st.info("ℹ️ Nenhum sinal ou convergência encontrada para os ativos selecionados.")
            
            if erros:
                with st.expander(f"⚠️ {len(erros)} Erro(s) ou Alerta(s) durante o screener"):
                    for erro in erros:
                        st.caption(erro)
    
    # ==========================================
    # Listas de Sinais e Convergências
    # ==========================================
    st.markdown("---")
    st.subheader("🎯 Sinais Acionáveis")
    
    if st.session_state.cacaus_sinais_acionaveis:
        for opp in st.session_state.cacaus_sinais_acionaveis:
            direcao_cor = "🟢" if opp['direcao'] == 'COMPRA' else "🔴"
            if st.button(
                f"{direcao_cor} {opp['ticker']} ({opp['tipo_sinal']})",
                key=f"btn_sinal_{opp['ticker']}",
                use_container_width=True,
                help=f"Entrada: R$ {opp['entrada']:.2f} | Stop: R$ {opp['stop']:.2f} | Alvo: R$ {opp['alvo']:.2f}"
            ):
                st.session_state.ativo_visualizar = opp['ticker']
                st.rerun()
    else:
        st.caption("Nenhum sinal acionável encontrado. Execute o screener.")

    st.markdown("---")
    st.subheader("🔍 Convergências Detectadas (Geral)")

    if st.session_state.cacaus_convergencias_gerais:
        # Filtrar para mostrar apenas as convergências que *não* são sinais acionáveis
        # para evitar duplicação visual se o usuário só quer ver as "outras"
        sinais_acionaveis_tickers = {s['ticker'] for s in st.session_state.cacaus_sinais_acionaveis}
        convergencias_nao_acionaveis = [
            c for c in st.session_state.cacaus_convergencias_gerais 
            if c['ticker'] not in sinais_acionaveis_tickers or c['tipo_convergencia'] == 'RECENTE'
        ]
        
        if convergencias_nao_acionaveis:
            for conv in convergencias_nao_acionaveis:
                direcao_cor = "🟢" if conv['direcao'] == 'COMPRA' else "🔴"
                help_text = f"Diário: {conv['barra_diario']} barra(s) | Semanal: {conv['barra_semanal']} barra(s)"
                if st.button(
                    f"{direcao_cor} {conv['ticker']} (Convergência {conv['tipo_convergencia']})",
                    key=f"btn_conv_{conv['ticker']}_{conv['tipo_convergencia']}",
                    use_container_width=True,
                    help=help_text
                ):
                    st.session_state.ativo_visualizar = conv['ticker']
                    st.rerun()
        else:
            st.caption("Todas as convergências são também sinais acionáveis ou não há convergências gerais.")
    else:
        st.caption("Nenhuma convergência detectada. Execute o screener.")


# ==========================================
# COLUNA DIREITA: GRÁFICOS
# ==========================================

with col_direita:
    
    st.subheader("📈 Gráficos do Indicador (Diário e Semanal)")
    
    if st.session_state.cacaus_todos_dados:
        
        ativos_disponiveis = sorted(list(st.session_state.cacaus_todos_dados.keys()))
        
        # Usar ativo do session_state ou primeiro da lista
        ativo_padrao = st.session_state.get('ativo_visualizar', ativos_disponiveis[0])
        
        if ativo_padrao not in ativos_disponiveis:
            ativo_padrao = ativos_disponiveis[0]
        
        ativo_selecionado = st.selectbox(
            "Selecione o Ativo para Visualizar Gráficos",
            options=ativos_disponiveis,
            index=ativos_disponiveis.index(ativo_padrao) if ativo_padrao in ativos_disponiveis else 0
        )
        
        dados_ativo = st.session_state.cacaus_todos_dados[ativo_selecionado]
        
        # Verificar se tem sinal acionável para exibir informações detalhadas
        sinal_acionavel_selecionado = next(
            (s for s in st.session_state.cacaus_sinais_acionaveis if s['ticker'] == ativo_selecionado),
            None
        )
        
        # Exibir status de convergência/sinal
        if sinal_acionavel_selecionado:
            st.success(f"🎯 SINAL ACIONÁVEL: {sinal_acionavel_selecionado['direcao']} ({sinal_acionavel_selecionado['tipo_sinal']})")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                direcao_cor = "🟢" if sinal_acionavel_selecionado['direcao'] == 'COMPRA' else "🔴"
                st.metric("Direção", f"{direcao_cor} {sinal_acionavel_selecionado['direcao']}")
            
            with col2:
                st.metric("Entrada", f"R$ {sinal_acionavel_selecionado['entrada']:.2f}")
            
            with col3:
                st.metric("Stop", f"R$ {sinal_acionavel_selecionado['stop']:.2f}")
            
            with col4:
                st.metric("Alvo", f"R$ {sinal_acionavel_selecionado['alvo']:.2f}")
            
            with col5:
                st.metric("R:R", sinal_acionavel_selecionado['rr'])
        
        else:
            # Não é um sinal acionável, mas pode ser uma convergência geral ou apenas cruzamentos
            conv = dados_ativo['convergencia']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if conv['cruzamento_diario']:
                    cor = "🟢" if conv['cruzamento_diario'] == 'COMPRA' else "🔴"
                    barras = f"({conv['barra_cruzamento_diario']} barra(s) atrás)"
                    st.info(f"📅 Diário: {cor} {conv['cruzamento_diario']} {barras}")
                else:
                    st.warning("📅 Diário: Sem cruzamento recente")
            
            with col2:
                if conv['cruzamento_semanal']:
                    cor = "🟢" if conv['cruzamento_semanal'] == 'COMPRA' else "🔴"
                    barras = f"({conv['barra_cruzamento_semanal']} barra(s) atrás)"
                    st.info(f"📆 Semanal: {cor} {conv['cruzamento_semanal']} {barras}")
                else:
                    st.warning("📆 Semanal: Sem cruzamento recente")
            
            with col3:
                if conv['convergente']:
                    st.success(f"✅ Convergente ({conv['tipo_sinal']})")
                else:
                    st.error("❌ Sem convergência")
        
        st.markdown("---")
        
        # Exibir ambos os gráficos lado a lado
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Gráfico Diário")
            fig_diario = criar_grafico_cacaus_channel(
                dados_ativo['df_diario'],
                dados_ativo['df_semanal'], # Passar ambos, mas a função usa o correto internamente
                ativo_selecionado,
                "Diário",
                num_barras_grafico
            )
            if fig_diario:
                st.plotly_chart(fig_diario, use_container_width=True)
            else:
                st.warning(f"Não foi possível gerar o gráfico diário para {ativo_selecionado}. Dados insuficientes.")

        with chart_col2:
            st.markdown("##### Gráfico Semanal")
            fig_semanal = criar_grafico_cacaus_channel(
                dados_ativo['df_diario'],
                dados_ativo['df_semanal'], # Passar ambos, mas a função usa o correto internamente
                ativo_selecionado,
                "Semanal",
                num_barras_grafico // 2 # Semanal geralmente tem menos barras
            )
            if fig_semanal:
                st.plotly_chart(fig_semanal, use_container_width=True)
            else:
                st.warning(f"Não foi possível gerar o gráfico semanal para {ativo_selecionado}. Dados insuficientes.")
    
    else:
        st.info("👈 Execute o screener na barra lateral para visualizar os gráficos ou use a análise individual abaixo.")
        
        # Permitir visualizar ativo individual sem screener
        st.markdown("---")
        st.subheader("🔍 Análise Individual Rápida")
        
        ticker_individual = st.text_input("Digite um ticker para análise rápida (ex: PETR4)", value="PETR4")
        
        if st.button("📊 Visualizar Ativo Individual", use_container_width=True):
            
            if not ticker_individual:
                st.error("Por favor, digite um ticker.")
            else:
                with st.spinner(f"Carregando dados de {ticker_individual}..."):
                    
                    try:
                        df_ativo = obter_dados_historicos_completos(ticker_individual, data_inicio, data_fim_dt)
                        
                        min_bars_needed = max(periodo_superior, periodo_inferior, ema_periodo) + lookback_cruzamento + 5
                        if not df_ativo.empty and len(df_ativo) >= min_bars_needed:
                            
                            df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
                            df_semanal_raw = resample_para_semanal(df_ativo)
                            
                            if not df_semanal_raw.empty and len(df_semanal_raw) >= min_bars_needed / 5:
                                df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                                
                                convergencia = detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback_cruzamento)
                                
                                st.session_state.cacaus_todos_dados = {
                                    ticker_individual: {
                                        'df_diario': df_diario,
                                        'df_semanal': df_semanal,
                                        'convergencia': convergencia
                                    }
                                }
                                # Limpar listas de sinais/convergências para não misturar com o screener
                                st.session_state.cacaus_sinais_acionaveis = []
                                st.session_state.cacaus_convergencias_gerais = []

                                if convergencia['convergente']:
                                    convergencias_gerais_temp = [{
                                        'ticker': ticker_individual,
                                        'direcao': convergencia['direcao'],
                                        'tipo_convergencia': convergencia['tipo_sinal'],
                                        'barra_diario': convergencia['barra_cruzamento_diario'],
                                        'barra_semanal': convergencia['barra_cruzamento_semanal']
                                    }]
                                    st.session_state.cacaus_convergencias_gerais = convergencias_gerais_temp

                                    if convergencia['tipo_sinal'] in ['SIMULTÂNEO', 'REENTRADA DIÁRIO', 'REENTRADA SEMANAL']:
                                        pontos = calcular_entrada_stop_alvo(df_diario, convergencia['direcao'], rr_ratio)
                                        sinais_acionaveis_temp = [{
                                            'ticker': ticker_individual,
                                            'direcao': convergencia['direcao'],
                                            'entrada': pontos['entrada'],
                                            'stop': pontos['stop'],
                                            'alvo': pontos['alvo'],
                                            'rr': pontos['rr'],
                                            'tipo_sinal': convergencia['tipo_sinal']
                                        }]
                                        st.session_state.cacaus_sinais_acionaveis = sinais_acionaveis_temp


                                st.session_state.ativo_visualizar = ticker_individual
                                
                                st.success(f"✅ Dados de {ticker_individual} carregados com sucesso!")
                                st.rerun()
                            else:
                                st.error("❌ Erro ao criar timeframe semanal ou dados insuficientes para o semanal.")
                        
                        else:
                            st.error(f"❌ Dados insuficientes para {ticker_individual} no período solicitado. Tente um ticker diferente ou período mais curto.")
                    
                    except Exception as e:
                        st.error(f"❌ Erro ao carregar {ticker_individual}: {str(e)}. Verifique se o ticker está correto.")


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")

with st.expander("📖 Como funciona o Cacau's Channel?"):
    st.markdown("""
    ### Estrutura do Indicador
    
    O **Cacau's Channel** é composto por quatro elementos principais que trabalham em conjunto para identificar oportunidades de trading:
    
    **Componentes Visuais:**
    
    A **Linha Superior (Vermelha)** representa a máxima dos últimos períodos configurados, funcionando como resistência dinâmica e indicando zonas de potencial reversão ou rompimento. Quando o preço se aproxima desta linha, sinaliza possível topo de movimento.
    
    A **Linha Inferior (Verde)** mostra a mínima dos últimos períodos, atuando como suporte dinâmico. Proximidade a esta linha sugere possível fundo de movimento ou zona de compra.
    
    A **Linha Média (Branca)** é calculada como ponto médio entre as linhas superior e inferior, representando o equilíbrio do canal. Sua posição relativa à EMA determina a tendência atual do ativo.
    
    A **EMA da Média (Laranja Tracejada)** suaviza os movimentos da linha média através de média móvel exponencial, fornecendo referência mais estável para identificação de tendências.
    
    ### Lógica de Sinais (Cruzamentos)
    
    O sistema detecta cruzamentos entre a Linha Média (branca) e a EMA (laranja) em dois timeframes diferentes para confirmar sinais.
    
    **Sinal de COMPRA ocorre quando:**
    - ✅ No timeframe **Semanal**, a Linha Branca cruza para CIMA da Linha Laranja
    - ✅ No timeframe **Diário**, a Linha Branca cruza para CIMA da Linha Laranja
    - ✅ Há **convergência**: ambos os cruzamentos apontam na mesma direção
    
    **Sinal de VENDA ocorre quando:**
    - ✅ No timeframe **Semanal**, a Linha Branca cruza para BAIXO da Linha Laranja
    - ✅ No timeframe **Diário**, a Linha Branca cruza para BAIXO da Linha Laranja
    - ✅ Há **convergência**: ambos os cruzamentos apontam na mesma direção
    
    ### Tipos de Sinal
    
    O sistema classifica os sinais de acordo com o timing dos cruzamentos:
    
    **SIMULTÂNEO** indica que ambos os timeframes (diário e semanal) apresentaram cruzamento na última barra, representando o sinal mais forte e recente.
    
    **REENTRADA DIÁRIO** ocorre quando o timeframe semanal já posicionado, e o diário acabou de cruzar, oferecendo oportunidade de entrada em tendência já estabelecida no prazo maior.
    
    **REENTRADA SEMANAL** acontece quando o diário já posicionado, e o semanal acabou de cruzar, confirmando a tendência de curto prazo com movimento de longo prazo.
    
    **RECENTE** identifica situações onde ambos os cruzamentos ocorreram há poucas barras (dentro do lookback configurado), mas não simultaneamente e nem na última barra de um dos timeframes.
    
    ### Gestão de Risco
    
    O sistema calcula automaticamente pontos de entrada, stop loss e alvo baseados na estrutura do canal:
    
    Para operações de **COMPRA**, o stop loss é posicionado na Linha Inferior (verde), protegendo contra rompimento do suporte. O alvo é calculado projetando a distância entre entrada e stop multiplicada pelo Risk/Reward configurado acima da entrada.
    
    Para operações de **VENDA**, o stop loss é posicionado na Linha Superior (vermelha), protegendo contra rompimento da resistência. O alvo é calculado projetando a distância entre stop e entrada multiplicada pelo Risk/Reward configurado abaixo da entrada.
    
    ### Parâmetros Configuráveis
    
    **Período Superior** controla a janela de cálculo da linha superior (resistência). Valores maiores criam canal mais amplo e estável.
    
    **Período Inferior** define a janela de cálculo da linha inferior (suporte). Pode ser diferente do superior para assimetria intencional.
    
    **EMA Período** determina a suavização da linha média. Valores menores tornam o indicador mais responsivo, valores maiores reduzem ruído.
    
    **Risk/Reward** estabelece a proporção entre risco assumido (distância até stop) e ganho esperado (distância até alvo).
    
    **Lookback Cruzamento** define quantas barras olhar para trás ao detectar cruzamentos, permitindo capturar sinais recentes mas não apenas da última barra.
    
    ### Interpretação dos Gráficos
    
    No timeframe **Diário**, você visualiza movimentos de curto prazo com maior granularidade, ideal para timing preciso de entrada e acompanhamento intraday da operação.
    
    No timeframe **Semanal**, você observa a tendência de médio prazo, essencial para confirmar a direção principal do movimento e evitar operações contra a tendência maior.
    
    A convergência entre ambos os timeframes aumenta significativamente a probabilidade de sucesso, pois indica alinhamento entre diferentes perspectivas temporais do mercado.
    
    ### Limitações e Avisos
    
    ⚠️ **Este é um sistema de análise técnica e não garante lucros**. Mercados financeiros envolvem risco de perda de capital.
    
    ⚠️ **Sinais falsos podem ocorrer**, especialmente em mercados laterais ou de baixa volatilidade. Use sempre stop loss.
    
    ⚠️ **A qualidade dos dados históricos impacta os resultados**. Períodos muito longos podem ter gaps ou inconsistências.
    
    ⚠️ **Não é recomendação de investimento**. Esta ferramenta serve apenas para análise técnica educacional. Consulte profissionais certificados antes de investir.
    
    ### Dicas de Uso
    
    Para melhores resultados, combine o Cacau's Channel com análise fundamentalista, considere o contexto macroeconômico e notícias do setor, e sempre opere com gestão de risco adequada ao seu perfil.
    
    Teste diferentes combinações de parâmetros em períodos históricos para entender o comportamento do indicador antes de usar em operações reais.
    
    Priorize sinais do tipo SIMULTÂNEO em ativos com boa liquidez e volume consistente.
    """)
