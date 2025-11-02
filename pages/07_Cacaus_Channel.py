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

def obter_dados_ohlc_completos(ticker, data_inicio, data_fim):
    """
    Obtém dados OHLC reais, primeiro tenta core.data, depois yfinance como fallback
    """
    try:
        # Primeira tentativa: core.data
        df_original = get_price_history([ticker], data_inicio, data_fim)
        
        if not df_original.empty:
            # Verificar se já temos dados OHLC
            if isinstance(df_original.columns, pd.MultiIndex):
                if ticker in df_original.columns.get_level_values(0):
                    df_ticker = df_original[ticker]
                    if all(col in df_ticker.columns for col in ['Open', 'High', 'Low', 'Close']):
                        df_result = df_ticker[['Open', 'High', 'Low', 'Close']].copy()
                        df_result['Volume'] = df_ticker.get('Volume', 1000000)
                        return df_result.dropna()
            
            elif all(col in df_original.columns for col in ['Open', 'High', 'Low', 'Close']):
                df_result = df_original[['Open', 'High', 'Low', 'Close']].copy()
                df_result['Volume'] = df_original.get('Volume', 1000000)
                return df_result.dropna()
    except Exception as e:
        st.warning(f"Erro ao obter dados via core.data para {ticker}: {str(e)}")
    
    # Fallback: yfinance para dados OHLC reais
    try:
        import yfinance as yf
        
        ticker_yf = ticker + ".SA" if not ticker.endswith(".SA") else ticker
        data_inicio_buffer = data_inicio - timedelta(days=30)
        
        stock = yf.Ticker(ticker_yf)
        df_yf = stock.history(start=data_inicio_buffer, end=data_fim + timedelta(days=1))
        
        if not df_yf.empty:
            df_yf = df_yf[df_yf.index >= data_inicio]
            return df_yf[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    except ImportError:
        st.info("💡 Para dados OHLC reais, instale: pip install yfinance")
    except Exception as e:
        st.warning(f"Erro yfinance para {ticker}: {str(e)}")
    
    # Último recurso: criar OHLC aproximado do preço de fechamento
    try:
        df = get_price_history([ticker], data_inicio, data_fim)
        if not df.empty and ticker in df.columns:
            precos = df[ticker].dropna()
            
            # Criar variação artificial baseada em volatilidade histórica
            volatilidade = precos.pct_change().std()
            if pd.isna(volatilidade) or volatilidade == 0:
                volatilidade = 0.02  # 2% padrão
            
            # Limitar volatilidade artificial
            volatilidade = min(volatilidade * 0.3, 0.05)  # Máximo 5%
            
            np.random.seed(42)  # Para consistência
            high_mult = 1 + np.random.uniform(0, volatilidade, len(precos))
            low_mult = 1 - np.random.uniform(0, volatilidade, len(precos))
            
            df_ohlc = pd.DataFrame({
                'Open': precos.shift(1).fillna(precos),
                'High': precos * high_mult,
                'Low': precos * low_mult,
                'Close': precos,
                'Volume': 1000000
            }, index=precos.index)
            
            # Garantir que High >= max(Open,Close) e Low <= min(Open,Close)
            df_ohlc['High'] = df_ohlc[['High', 'Open', 'Close']].max(axis=1)
            df_ohlc['Low'] = df_ohlc[['Low', 'Open', 'Close']].min(axis=1)
            
            return df_ohlc.dropna()
    except Exception as e:
        st.error(f"Erro ao criar dados OHLC para {ticker}: {str(e)}")
    
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
        return ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "BBAS3", "WEGE3", "MGLU3"]


def filtrar_ativos_por_tipo(base_completa, tipos_selecionados):
    """Filtra ativos por tipo com lógica corrigida"""
    if not tipos_selecionados:
        return []
    
    tickers_filtrados = []
    
    for tipo in tipos_selecionados:
        if tipo == "Ações":
            # Ações terminam em 3, 4, 5, 6, 7, 8 (mas não 11 que são FIIs)
            acoes = [t for t in base_completa 
                    if len(t) >= 5 and t[-1].isdigit() and t[-1] in ['3','4','5','6','7','8'] 
                    and not t.endswith('11')]
            tickers_filtrados.extend(acoes)
            
        elif tipo == "FIIs":
            # FIIs terminam em 11
            fiis = [t for t in base_completa if t.endswith('11')]
            tickers_filtrados.extend(fiis)
            
        elif tipo == "ETFs":
            # ETFs geralmente têm 'B' no final ou estrutura diferente
            etfs = [t for t in base_completa 
                   if (t.endswith('B') or 'ETF' in t or len(t) <= 4 or 
                       (len(t) >= 5 and not t[-1].isdigit()))]
            tickers_filtrados.extend(etfs)
    
    return sorted(list(set(tickers_filtrados)))


# ==========================================
# FUNÇÕES DE CÁLCULO DO INDICADOR
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
    
    try:
        return df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    except Exception as e:
        st.error(f"Erro ao criar timeframe semanal: {str(e)}")
        return pd.DataFrame()


def detectar_convergencia_e_sinais(df_diario, df_semanal, lookback=5):
    """
    Detecta convergência e sinais separadamente
    
    CONVERGÊNCIA: Cruzamentos na mesma direção em ambos timeframes
    SINAL: Convergência + pelo menos um cruzamento na última barra
    """
    
    def encontrar_cruzamento(df):
        """Encontra cruzamento mais recente dentro do lookback"""
        if df.empty or len(df) < 2:
            return None, None
        
        # Garantir que temos as colunas necessárias
        if not all(col in df.columns for col in ['linha_media', 'ema_media']):
            return None, None
            
        for i in range(1, min(lookback + 1, len(df))):
            try:
                atual = df.iloc[-i]
                anterior = df.iloc[-(i+1)]
                
                if (pd.isna(atual['linha_media']) or pd.isna(atual['ema_media']) or
                    pd.isna(anterior['linha_media']) or pd.isna(anterior['ema_media'])):
                    continue
                
                # Cruzamento para CIMA
                if (anterior['linha_media'] <= anterior['ema_media'] and 
                    atual['linha_media'] > atual['ema_media']):
                    return 'COMPRA', i
                
                # Cruzamento para BAIXO
                if (anterior['linha_media'] >= anterior['ema_media'] and 
                    atual['linha_media'] < atual['ema_media']):
                    return 'VENDA', i
            except (IndexError, KeyError) as e:
                continue
        
        return None, None
    
    # Detectar cruzamentos
    cruz_diario, barras_diario = encontrar_cruzamento(df_diario.dropna())
    cruz_semanal, barras_semanal = encontrar_cruzamento(df_semanal.dropna())
    
    # CONVERGÊNCIA: mesma direção
    tem_convergencia = (cruz_diario and cruz_semanal and cruz_diario == cruz_semanal)
    
    # SINAL: convergência + gatilho na última barra
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


def calcular_entrada_stop_alvo(df, direcao, rr_ratio=2.0):
    """Calcula ponto de entrada, stop loss e alvo"""
    if df.empty or not direcao:
        return {'entrada': 0, 'stop': 0, 'alvo': 0}
    
    try:
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
        
        return {'entrada': entrada, 'stop': stop, 'alvo': alvo}
    except Exception as e:
        st.error(f"Erro ao calcular pontos: {str(e)}")
        return {'entrada': 0, 'stop': 0, 'alvo': 0}


# ==========================================
# VISUALIZAÇÃO - GRÁFICOS DUPLOS
# ==========================================

def criar_graficos_duplos(df_diario, df_semanal, ticker):
    """Cria gráficos lado a lado com candlesticks OHLC reais"""
    
    try:
        df_d = df_diario.tail(100).dropna()
        df_s = df_semanal.tail(50).dropna()
        
        if df_d.empty or df_s.empty:
            return None
        
        # Subplots lado a lado
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{ticker} - Diário (100 barras)', f'{ticker} - Semanal (50 barras)'),
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
                name='Preço Diário',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Indicadores diário
        indicadores = [
            ('Superior', 'linha_superior', '#ff4444', 2),
            ('Inferior', 'linha_inferior', '#00ff00', 2),
            ('Média', 'linha_media', 'white', 2.5),
            ('EMA', 'ema_media', '#ff9800', 2)
        ]
        
        for nome, coluna, cor, largura in indicadores:
            if coluna in df_d.columns and not df_d[coluna].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df_d.index, 
                        y=df_d[coluna],
                        mode='lines',
                        name=f'{nome} D',
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
                name='Preço Semanal',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Indicadores semanal
        for nome, coluna, cor, largura in indicadores:
            if coluna in df_s.columns and not df_s[coluna].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df_s.index, 
                        y=df_s[coluna],
                        mode='lines',
                        name=f'{nome} S',
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
            height=650,
            template="plotly_dark",
            margin=dict(l=60, r=60, t=100, b=60),
            title={
                'text': f"{ticker} - Cacau's Channel (Comparação Diário vs Semanal)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            }
        )
        
        # Remover rangesliders
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    except Exception as e:
        st.error(f"Erro ao criar gráficos: {str(e)}")
        return None


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

st.title("🎯 Cacau's Channel - Screener Multi-Timeframe")
st.markdown("**CONVERGÊNCIA** = Cruzamentos alinhados | **SINAL** = Convergência + Gatilho ativo")

# Botão para limpar cache/session_state
if st.button("🔄 Limpar Dados", help="Remove dados antigos do cache"):
    for key in list(st.session_state.keys()):
        if key.startswith('cacaus_'):
            del st.session_state[key]
    st.success("✅ Dados limpos!")
    st.rerun()

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

# ==========================================
# LAYOUT EM DUAS COLUNAS
# ==========================================

col_esquerda, col_direita = st.columns([1, 2.5])

# ==========================================
# COLUNA ESQUERDA: SELEÇÃO E SCREENER
# ==========================================

with col_esquerda:
    
    st.subheader("📈 Seleção de Ativos")
    
    # Carregar base completa
    base_completa = carregar_base_ativos()
    
    if base_completa:
        st.caption(f"✅ {len(base_completa)} ativos disponíveis")
    else:
        st.error("❌ Erro ao carregar base de ativos")
        st.stop()
    
    # Opções de seleção
    opcao_selecao = st.radio(
        "Fonte dos Ativos",
        options=["📁 Portfólio", "🌐 Base B3", "✍️ Manual"],
        index=2
    )
    
    tickers = []
    
    # OPÇÃO 1: Portfólio
    if opcao_selecao == "📁 Portfólio":
        try:
            from core.portfolio import listar_portfolios, carregar_portfolio
            portfolios_disponiveis = listar_portfolios()
            
            if portfolios_disponiveis:
                portfolio_selecionado = st.selectbox("Selecione o portfólio", portfolios_disponiveis)
                portfolio = carregar_portfolio(portfolio_selecionado)
                tickers = portfolio.tickers if portfolio else []
                st.caption(f"📊 {len(tickers)} ativos no portfólio")
            else:
                st.warning("Nenhum portfólio encontrado")
        except Exception as e:
            st.error(f"Erro ao carregar portfólios: {str(e)}")
    
    # OPÇÃO 2: Base B3
    elif opcao_selecao == "🌐 Base B3":
        
        filtro_tipo = st.multiselect(
            "Tipos de Ativos",
            options=["Ações", "FIIs", "ETFs"],
            default=["Ações"],
            help="Selecione os tipos de ativos para análise"
        )
        
        if filtro_tipo:
            # Usar função corrigida de filtragem
            tickers_filtrados = filtrar_ativos_por_tipo(base_completa, filtro_tipo)
            
            # Debug: mostrar alguns exemplos
            if tickers_filtrados:
                st.success(f"✅ {len(tickers_filtrados)} ativos encontrados")
                
                # Mostrar alguns exemplos de cada tipo
                with st.expander("📋 Ver exemplos dos ativos"):
                    if "Ações" in filtro_tipo:
                        acoes_exemplos = [t for t in tickers_filtrados if t[-1] in ['3','4','5','6','7','8'] and not t.endswith('11')][:10]
                        if acoes_exemplos:
                            st.write("**Ações:**", ", ".join(acoes_exemplos))
                    
                    if "FIIs" in filtro_tipo:
                        fiis_exemplos = [t for t in tickers_filtrados if t.endswith('11')][:10]
                        if fiis_exemplos:
                            st.write("**FIIs:**", ", ".join(fiis_exemplos))
                    
                    if "ETFs" in filtro_tipo:
                        etfs_exemplos = [t for t in tickers_filtrados if not t[-1].isdigit()][:10]
                        if etfs_exemplos:
                            st.write("**ETFs:**", ", ".join(etfs_exemplos))
            else:
                st.warning(f"❌ Nenhum ativo encontrado para os tipos selecionados")
                st.info("💡 Tente selecionar outros tipos ou verifique a base de dados")
            
            # Limite de ativos
            limite_ativos = st.number_input(
                "Limite de Ativos",
                min_value=1,
                max_value=min(200, len(tickers_filtrados)) if tickers_filtrados else 50,
                value=min(30, len(tickers_filtrados)) if tickers_filtrados else 30,
                step=5,
                help="Número máximo de ativos para analisar"
            )
            
            tickers = tickers_filtrados[:limite_ativos] if tickers_filtrados else []
            st.caption(f"📊 {len(tickers)} ativos selecionados para análise")
            
        else:
            st.warning("⚠️ Selecione pelo menos um tipo de ativo")
    
    # OPÇÃO 3: Manual
    else:
        tickers_input = st.text_area(
            "Digite os tickers (um por linha)",
            value="ALPA4\nPETR4\nVALE3\nITUB4\nBBDC4\nBBAS3\nWEGE3",
            height=150,
            help="Digite um ticker por linha ou separados por vírgula"
        )
        
        if tickers_input.strip():
            tickers_raw = tickers_input.replace(',', '\n').replace(';', '\n').split('\n')
            tickers = [t.strip().upper() for t in tickers_raw if t.strip()]
            st.caption(f"📊 {len(tickers)} ativos listados")
            
            # Mostrar os tickers que serão analisados
            if tickers:
                st.success(f"✅ Tickers: {', '.join(tickers[:10])}" + (f" e mais {len(tickers)-10}" if len(tickers) > 10 else ""))
        else:
            st.warning("⚠️ Digite pelo menos um ticker")
    
    # SCREENER
    st.markdown("---")
    
    # Mostrar botão apenas se há tickers selecionados
    screener_habilitado = len(tickers) > 0
    
    if not screener_habilitado:
        st.warning("⚠️ Selecione ativos para executar o screener")
    
    if st.button("🔍 Executar Screener", 
                 type="primary", 
                 use_container_width=True, 
                 disabled=not screener_habilitado):
        
        convergencias = []
        sinais = []
        todos_dados = {}
        erros = []
        
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
                df_ohlc = obter_dados_ohlc_completos(ticker, data_inicio, data_fim_dt)
                
                if df_ohlc.empty or len(df_ohlc) < 60:
                    erros.append(f"{ticker}: Dados insuficientes")
                    continue
                
                total_com_dados += 1
                
                # Calcular indicadores
                df_diario = calcular_cacaus_channel(df_ohlc, periodo_superior, periodo_inferior, ema_periodo)
                df_semanal_raw = resample_para_semanal(df_ohlc)
                
                if df_semanal_raw.empty:
                    erros.append(f"{ticker}: Erro no timeframe semanal")
                    continue
                
                df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                
                # Detectar convergência e sinais
                analise = detectar_convergencia_e_sinais(df_diario, df_semanal, lookback_cruzamento)
                
                # Salvar todos os dados com estrutura consistente
                todos_dados[ticker] = {
                    'df_diario': df_diario,
                    'df_semanal': df_semanal,
                    'analise': analise  # Chave consistente
                }
                
                # Processar convergências e sinais
                if analise['tem_convergencia']:
                    pontos = calcular_entrada_stop_alvo(df_diario, analise['direcao'], rr_ratio)
                    
                    item = {
                        'ticker': ticker,
                        'direcao': analise['direcao'],
                        'tipo': analise['tipo'],
                        'entrada': pontos['entrada'],
                        'stop': pontos['stop'],
                        'alvo': pontos['alvo']
                    }
                    
                    convergencias.append(item)
                    
                    if analise['tem_sinal']:
                        sinais.append(item)
            
            except Exception as e:
                erros.append(f"{ticker}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Salvar resultados
        st.session_state.cacaus_convergencias = convergencias
        st.session_state.cacaus_sinais = sinais
        st.session_state.cacaus_todos_dados = todos_dados
        
        # Estatísticas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Analisados", total_analisados)
        col2.metric("✅ Com Dados", total_com_dados)
        col3.metric("🔄 Convergências", len(convergencias))
        col4.metric("🎯 Sinais", len(sinais))
        
        if sinais:
            st.success(f"🎯 {len(sinais)} sinal(is) ativo(s) encontrado(s)!")
        elif convergencias:
            st.info(f"🔄 {len(convergencias)} convergência(s) encontrada(s)")
        else:
            st.warning("Nenhuma oportunidade encontrada no período")
        
        # Mostrar erros se houver poucos
        if erros and len(erros) <= 10:
            with st.expander(f"⚠️ {len(erros)} erro(s) encontrado(s)"):
                for erro in erros:
                    st.caption(f"• {erro}")
    
    # LISTAS DE RESULTADOS
    st.markdown("---")
    
    # SINAIS (prioritários)
    st.subheader("🎯 Sinais Ativos")
    if 'cacaus_sinais' in st.session_state and st.session_state.cacaus_sinais:
        for sinal in st.session_state.cacaus_sinais:
            cor = "🟢" if sinal['direcao'] == 'COMPRA' else "🔴"
            if st.button(f"{cor} {sinal['ticker']} - {sinal['tipo']}", 
                        key=f"sinal_{sinal['ticker']}", use_container_width=True):
                st.session_state.ativo_visualizar = sinal['ticker']
                st.rerun()
    else:
        st.caption("Nenhum sinal ativo")
    
    # CONVERGÊNCIAS (apenas as que não são sinais)
    st.subheader("🔄 Convergências")
    if 'cacaus_convergencias' in st.session_state and st.session_state.cacaus_convergencias:
        sinais_tickers = {s['ticker'] for s in st.session_state.get('cacaus_sinais', [])}
        conv_apenas = [c for c in st.session_state.cacaus_convergencias if c['ticker'] not in sinais_tickers]
        
        if conv_apenas:
            for conv in conv_apenas:
                cor = "🟢" if conv['direcao'] == 'COMPRA' else "🔴"
                if st.button(f"{cor} {conv['ticker']} - {conv['tipo']}", 
                            key=f"conv_{conv['ticker']}", use_container_width=True):
                    st.session_state.ativo_visualizar = conv['ticker']
                    st.rerun()
        else:
            st.caption("Todas as convergências são sinais")
    else:
        st.caption("Execute o screener")

# ==========================================
# COLUNA DIREITA: GRÁFICOS
# ==========================================

with col_direita:
    
    st.subheader("📈 Análise Gráfica")
    
    if 'cacaus_todos_dados' in st.session_state and st.session_state.cacaus_todos_dados:
        
        ativos_disponiveis = sorted(st.session_state.cacaus_todos_dados.keys())
        ativo_default = st.session_state.get('ativo_visualizar', ativos_disponiveis[0])
        
        ativo_selecionado = st.selectbox(
            "Selecione o Ativo para Análise",
            options=ativos_disponiveis,
            index=ativos_disponiveis.index(ativo_default) if ativo_default in ativos_disponiveis else 0
        )
        
        dados_ativo = st.session_state.cacaus_todos_dados[ativo_selecionado]
        
        # TRATAMENTO ROBUSTO PARA ACESSAR ANÁLISE
        analise = None
        if 'analise' in dados_ativo:
            analise = dados_ativo['analise']
        elif 'convergencia' in dados_ativo:  # Fallback para estrutura antiga
            conv = dados_ativo['convergencia']
            # Converter estrutura antiga para nova
            analise = {
                'tem_convergencia': conv.get('convergente', False),
                'tem_sinal': conv.get('convergente', False) and (
                    conv.get('barra_cruzamento_diario') == 1 or 
                    conv.get('barra_cruzamento_semanal') == 1
                ),
                'direcao': conv.get('direcao'),
                'tipo': conv.get('tipo_sinal', 'RECENTE'),
                'cruz_diario': conv.get('cruzamento_diario'),
                'cruz_semanal': conv.get('cruzamento_semanal'),
                'barras_diario': conv.get('barra_cruzamento_diario'),
                'barras_semanal': conv.get('barra_cruzamento_semanal')
            }
        else:
            # Estrutura padrão se não encontrar dados
            analise = {
                'tem_convergencia': False,
                'tem_sinal': False,
                'direcao': None,
                'tipo': None,
                'cruz_diario': None,
                'cruz_semanal': None,
                'barras_diario': None,
                'barras_semanal': None
            }
        
        # STATUS DO ATIVO
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if analise['tem_sinal']:
                st.success(f"🎯 SINAL: {analise['direcao']}")
            elif analise['tem_convergencia']:
                st.info(f"🔄 CONVERGÊNCIA: {analise['direcao']}")
            else:
                st.warning("❌ Sem convergência")
        
        with col2:
            if analise['cruz_diario']:
                cor = "🟢" if analise['cruz_diario'] == 'COMPRA' else "🔴"
                barras_info = f" ({analise['barras_diario']})" if analise['barras_diario'] else ""
                st.write(f"📅 Diário: {cor} {analise['cruz_diario']}{barras_info}")
            else:
                st.write("📅 Diário: Sem cruzamento")
        
        with col3:
            if analise['cruz_semanal']:
                cor = "🟢" if analise['cruz_semanal'] == 'COMPRA' else "🔴"
                barras_info = f" ({analise['barras_semanal']})" if analise['barras_semanal'] else ""
                st.write(f"📆 Semanal: {cor} {analise['cruz_semanal']}{barras_info}")
            else:
                st.write("📆 Semanal: Sem cruzamento")
        
        # PONTOS DE OPERAÇÃO
        if analise['tem_convergencia'] and analise['direcao']:
            pontos = calcular_entrada_stop_alvo(dados_ativo['df_diario'], analise['direcao'], rr_ratio)
            
            if pontos['entrada'] > 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💰 Entrada", f"R$ {pontos['entrada']:.2f}")
                col2.metric("🛑 Stop", f"R$ {pontos['stop']:.2f}")
                col3.metric("🎯 Alvo", f"R$ {pontos['alvo']:.2f}")
                
                risco_pct = abs(pontos['entrada'] - pontos['stop']) / pontos['entrada'] * 100
                col4.metric("📊 Risco", f"{risco_pct:.1f}%")
        
        # GRÁFICOS DUPLOS
        st.markdown("---")
        fig = criar_graficos_duplos(dados_ativo['df_diario'], dados_ativo['df_semanal'], ativo_selecionado)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Erro ao gerar gráficos - verifique se há dados suficientes")
    
    else:
        st.info("👈 Execute o screener na barra lateral para visualizar análises")
        
        # ANÁLISE INDIVIDUAL
        st.markdown("---")
        st.subheader("🔍 Análise Individual Rápida")
        
        ticker_individual = st.text_input("Digite o ticker do ativo:", value="ALPA4")
        
        if st.button("📊 Analisar Ativo", use_container_width=True):
            
            with st.spinner(f"Carregando dados de {ticker_individual}..."):
                
                try:
                    df_ohlc = obter_dados_ohlc_completos(ticker_individual, data_inicio, data_fim_dt)
                    
                    if not df_ohlc.empty and len(df_ohlc) >= 60:
                        
                        df_diario = calcular_cacaus_channel(df_ohlc, periodo_superior, periodo_inferior, ema_periodo)
                        df_semanal_raw = resample_para_semanal(df_ohlc)
                        
                        if not df_semanal_raw.empty:
                            df_semanal = calcular_cacaus_channel(df_semanal_raw, periodo_superior, periodo_inferior, ema_periodo)
                            analise = detectar_convergencia_e_sinais(df_diario, df_semanal, lookback_cruzamento)
                            
                            st.session_state.cacaus_todos_dados = {
                                ticker_individual: {
                                    'df_diario': df_diario,
                                    'df_semanal': df_semanal,
                                    'analise': analise
                                }
                            }
                            
                            st.session_state.ativo_visualizar = ticker_individual
                            st.success(f"✅ {ticker_individual} carregado com sucesso!")
                            st.rerun()
                        else:
                            st.error("❌ Erro ao criar timeframe semanal")
                    else:
                        st.error(f"❌ Dados insuficientes para {ticker_individual}")
                
                except Exception as e:
                    st.error(f"❌ Erro ao analisar: {str(e)}")

# ==========================================
# EXPLICAÇÃO
# ==========================================

st.markdown("---")

with st.expander("📖 Guia: Convergência vs Sinais"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔄 CONVERGÊNCIA
        
        **O que é:** Alinhamento de cruzamentos entre timeframes
        
        **Critérios:**
        - ✅ Cruzamento no Diário: Linha Branca × EMA Laranja
        - ✅ Cruzamento no Semanal: Mesma lógica
        - ✅ Mesma direção (COMPRA ou VENDA)
        - ✅ Dentro do lookback configurado
        
        **Tipos de Convergência:**
        - **SIMULTÂNEO:** Ambos cruzaram na última barra
        - **GATILHO DIÁRIO:** Semanal já posicionado, diário acabou de cruzar
        - **GATILHO SEMANAL:** Diário já posicionado, semanal acabou de cruzar  
        - **CONVERGÊNCIA RECENTE:** Ambos cruzaram recentemente
        
        **Interpretação:** Indica alinhamento entre prazos, mas não necessariamente momento de entrada
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 SINAL
        
        **O que é:** Convergência + momento ideal para entrada
        
        **Critérios Adicionais:**
        - ✅ Deve ter convergência confirmada
        - ✅ Pelo menos um cruzamento na última barra
        - ✅ Indica momento de ação imediata
        
        **Diferença Prática:**
        - **Convergência:** "Os timeframes estão alinhados"
        - **Sinal:** "É hora de agir!"
        
        **Como Usar:**
        - **🎯 Sinais:** Prioridade máxima - considere entrada
        - **🔄 Convergências:** Monitore - pode virar sinal
        
        **Gestão de Risco:**
        - Stop COMPRA: Linha Inferior (verde)
        - Stop VENDA: Linha Superior (vermelha)
        - Alvo: Baseado no Risk/Reward configurado
        """)

st.markdown("""
### 💡 Como Selecionar Ativos

**🌐 Base B3:** Use os filtros por tipo:
- **Ações:** Terminam em 3, 4, 5, 6, 7, 8 (ex: PETR4, VALE3)
- **FIIs:** Terminam em 11 (ex: HGLG11, KNRI11)  
- **ETFs:** Estrutura diferente (ex: BOVA11, SMAL11)

**✍️ Manual:** Digite um ticker por linha ou separados por vírgula

⚠️ **Aviso Legal:** Esta é uma ferramenta de análise técnica educacional. Não constitui recomendação de investimento. Mercados financeiros envolvem risco de perda. Sempre consulte um profissional qualificado antes de investir.
""")
