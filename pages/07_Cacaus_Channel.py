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
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# ==========================================

def criar_ohlc_correto(df_precos, ticker):
    """
    Cria DataFrame OHLC correto a partir dos dados de preços históricos
    Assume que df_precos contém dados diários completos
    """
    try:
        # Se o DataFrame já tem colunas OHLC
        if all(col in df_precos.columns for col in ['Open', 'High', 'Low', 'Close']):
            df_ohlc = df_precos[['Open', 'High', 'Low', 'Close']].copy()
            if 'Volume' in df_precos.columns:
                df_ohlc['Volume'] = df_precos['Volume']
            else:
                df_ohlc['Volume'] = 0
            return df_ohlc.dropna()
        
        # Se temos apenas uma coluna de preços (Close)
        if ticker in df_precos.columns:
            precos = df_precos[ticker].dropna()
            
            # Criar OHLC artificial baseado no Close
            # Isso é uma aproximação quando não temos dados OHLC reais
            df_ohlc = pd.DataFrame({
                'Open': precos,
                'High': precos * 1.005,  # Aproximação: High ~0.5% acima
                'Low': precos * 0.995,   # Aproximação: Low ~0.5% abaixo
                'Close': precos,
                'Volume': 0
            }, index=precos.index)
            
            return df_ohlc
        
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Erro ao criar OHLC para {ticker}: {str(e)}")
        return pd.DataFrame()


def obter_dados_historicos_completos(ticker, data_inicio, data_fim, max_tentativas=3):
    """
    Obtém dados históricos com múltiplas tentativas e validação
    """
    for tentativa in range(max_tentativas):
        try:
            # Adicionar margem de segurança nas datas
            data_inicio_ajustada = data_inicio - timedelta(days=30)
            
            df = get_price_history([ticker], data_inicio_ajustada, data_fim)
            
            if df.empty:
                continue
            
            # Criar OHLC correto
            df_ohlc = criar_ohlc_correto(df, ticker)
            
            if len(df_ohlc) >= 50:  # Mínimo de 50 barras
                # Filtrar para o período solicitado
                df_ohlc = df_ohlc[df_ohlc.index >= data_inicio]
                return df_ohlc
            
        except Exception as e:
            if tentativa == max_tentativas - 1:
                st.warning(f"Erro ao obter dados de {ticker}: {str(e)}")
            continue
    
    return pd.DataFrame()


# ==========================================
# FUNÇÕES DE CÁLCULO DO INDICADOR
# ==========================================

def calcular_cacaus_channel(df, periodo_superior=20, periodo_inferior=30, ema_periodo=9):
    """Calcula o indicador Cacau's Channel"""
    df = df.copy()
    
    # Garantir que temos dados suficientes
    if len(df) < max(periodo_superior, periodo_inferior, ema_periodo):
        return df
    
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
    except Exception as e:
        st.error(f"Erro ao criar timeframe semanal: {str(e)}")
        return pd.DataFrame()


def detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback=5):
    """
    Detecta convergência de CRUZAMENTOS entre timeframes
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
        idx_atual = -i
        idx_anterior = -(i+1)
        
        linha_media_atual = df_diario['linha_media'].iloc[idx_atual]
        ema_media_atual = df_diario['ema_media'].iloc[idx_atual]
        linha_media_anterior = df_diario['linha_media'].iloc[idx_anterior]
        ema_media_anterior = df_diario['ema_media'].iloc[idx_anterior]
        
        # Verificar se os valores são válidos
        if pd.isna(linha_media_atual) or pd.isna(ema_media_atual):
            continue
        
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
        idx_atual = -i
        idx_anterior = -(i+1)
        
        linha_media_atual = df_semanal['linha_media'].iloc[idx_atual]
        ema_media_atual = df_semanal['ema_media'].iloc[idx_atual]
        linha_media_anterior = df_semanal['linha_media'].iloc[idx_anterior]
        ema_media_anterior = df_semanal['ema_media'].iloc[idx_anterior]
        
        # Verificar se os valores são válidos
        if pd.isna(linha_media_atual) or pd.isna(ema_media_atual):
            continue
        
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


# ==========================================
# VISUALIZAÇÃO
# ==========================================

def criar_grafico_cacaus_channel(df_diario, df_semanal, ticker, timeframe_ativo="Diário"):
    """Cria gráfico do Cacau's Channel com candlesticks completos e bem centralizado"""
    
    df = df_diario if timeframe_ativo == "Diário" else df_semanal
    
    # Determinar número de barras a mostrar
    num_barras = 100 if timeframe_ativo == "Diário" else 50
    df = df.tail(num_barras).copy()
    
    # Verificar se temos dados suficientes
    if df.empty or len(df) < 10:
        st.warning(f"Dados insuficientes para gerar gráfico de {ticker}")
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
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
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
        line=dict(color='#ff4444', width=2),
        hovertemplate='Superior: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Linha Inferior (verde)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['linha_inferior'],
        mode='lines',
        name='Linha Inferior',
        line=dict(color='#00ff00', width=2),
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
        line=dict(color='#ff9800', width=2.5, dash='dash'),
        hovertemplate='EMA: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Calcular range de preços para melhor centralização
    preco_min = df[['Low', 'linha_inferior']].min().min()
    preco_max = df[['High', 'linha_superior']].max().max()
    margem = (preco_max - preco_min) * 0.1  # 10% de margem
    
    fig.update_layout(
        title={
            'text': f"{ticker} - Cacau's Channel ({timeframe_ativo})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        height=700,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        
        # Melhorar centralização e margens
        margin=dict(l=80, r=80, t=100, b=80),
        
        # Configurar eixo Y para melhor visualização
        yaxis=dict(
            range=[preco_min - margem, preco_max + margem],
            autorange=False,
            fixedrange=False
        ),
        
        # Configurar eixo X
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date'
        ),
        
        # Legenda
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        
        # Cor de fundo
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    # Adicionar grid para melhor visualização
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#333333')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#333333')
    
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
    
    dias_periodo = periodos_dias[periodo_analise]
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
    
    st.subheader("📈 Ativos")
    
    # Carregar base completa
    base_completa = carregar_base_ativos()
    
    if base_completa:
        st.caption(f"✅ {len(base_completa)} ativos disponíveis")
    
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
                portfolio_selecionado = st.selectbox("Selecione o portfólio", portfolios_disponiveis, label_visibility="collapsed")
                portfolio = carregar_portfolio(portfolio_selecionado)
                tickers = portfolio.tickers if portfolio else []
                st.caption(f"📊 {len(tickers)} ativos no portfólio")
            else:
                st.warning("Nenhum portfólio encontrado")
        except Exception as e:
            st.error(f"Erro ao carregar portfólios: {str(e)}")
    
    # OPÇÃO 2: Base B3
    elif opcao_selecao == "🌐 Base B3":
        if base_completa:
            
            filtro_tipo = st.multiselect(
                "Tipo de Ativo",
                options=["Ações", "FIIs", "ETFs", "Todos"],
                default=["Ações"],
                label_visibility="collapsed"
            )
            
            limite_ativos = st.number_input(
                "Limite de Ativos",
                min_value=10,
                max_value=min(500, len(base_completa)),
                value=50,
                step=10,
                label_visibility="collapsed",
                help="Número máximo de ativos para analisar"
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
            
            st.caption(f"📊 {len(tickers)} ativos selecionados")
    
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
        
        st.caption(f"📊 {len(tickers)} ativos listados")
    
    # Botão de screener
    st.markdown("---")
    
    if st.button("🔍 Executar Screener", type="primary", use_container_width=True):
        
        if not tickers:
            st.error("❌ Nenhum ativo selecionado")
        else:
            oportunidades = []
            todos_dados = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_analisados = 0
            total_com_dados = 0
            total_convergentes = 0
            erros = []
            
            for idx, ticker in enumerate(tickers):
                
                progress = (idx + 1) / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"Analisando {ticker}... ({idx+1}/{len(tickers)})")
                
                total_analisados += 1
                
                try:
                    # Obter dados históricos completos
                    df_ativo = obter_dados_historicos_completos(ticker, data_inicio, data_fim_dt)
                    
                    if df_ativo.empty or len(df_ativo) < max(periodo_superior, periodo_inferior, ema_periodo) + 10:
                        erros.append(f"{ticker}: Dados insuficientes")
                        continue
                    
                    total_com_dados += 1
                    
                    # Calcular indicador no diário
                    df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
                    
                    # Criar timeframe semanal
                    df_semanal_raw = resample_para_semanal(df_ativo)
                    
                    if df_semanal_raw.empty or len(df_semanal_raw) < max(periodo_superior, periodo_inferior, ema_periodo) + 2:
                        erros.append(f"{ticker}: Erro ao criar semanal")
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
                    
                    # Adicionar apenas convergentes à lista de oportunidades
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
                    erros.append(f"{ticker}: {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            # Salvar resultados no session_state
            st.session_state.cacaus_oportunidades = oportunidades
            st.session_state.cacaus_todos_dados = todos_dados
            
            # Mostrar estatísticas
            st.metric("Analisados", total_analisados)
            st.metric("Com Dados", total_com_dados)
            st.metric("🎯 Sinais", total_convergentes)
            
            if oportunidades:
                st.success(f"✅ {len(oportunidades)} sinal(is) encontrado(s)!")
            else:
                st.info("ℹ️ Nenhum sinal convergente encontrado")
            
            if erros and len(erros) <= 10:
                with st.expander(f"⚠️ {len(erros)} erro(s)"):
                    for erro in erros:
                        st.caption(erro)
    
    # Mostrar lista de oportunidades
    st.markdown("---")
    st.subheader("🎯 Sinais Convergentes")
    
    if 'cacaus_oportunidades' in st.session_state and st.session_state.cacaus_oportunidades:
        
        oportunidades = st.session_state.cacaus_oportunidades
        
        for opp in oportunidades:
            direcao_cor = "🟢" if opp['direcao'] == 'COMPRA' else "🔴"
            
            if st.button(
                f"{direcao_cor} {opp['ticker']} - {opp['tipo_sinal']}",
                key=f"btn_{opp['ticker']}",
                use_container_width=True,
                help=f"Entrada: R$ {opp['entrada']:.2f} | Stop: R$ {opp['stop']:.2f} | Alvo: R$ {opp['alvo']:.2f}"
            ):
                st.session_state.ativo_visualizar = opp['ticker']
                st.rerun()
    else:
        st.caption("Execute o screener para ver sinais")


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
            "Selecione o Ativo",
            options=ativos_disponiveis,
            index=ativos_disponiveis.index(ativo_padrao) if ativo_padrao in ativos_disponiveis else 0
        )
        
        dados_ativo = st.session_state.cacaus_todos_dados[ativo_selecionado]
        
        # Verificar se tem sinal convergente
        opp_selecionada = None
        if 'cacaus_oportunidades' in st.session_state:
            opp_selecionada = next(
                (o for o in st.session_state.cacaus_oportunidades if o['ticker'] == ativo_selecionado),
                None
            )
        
        # Mostrar informações do sinal
        if opp_selecionada:
            # TEM SINAL CONVERGENTE
            st.success(f"🎯 SINAL CONVERGENTE: {opp_selecionada['direcao']}")
            
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
            # SEM SINAL CONVERGENTE - Mostrar status dos cruzamentos
            conv = dados_ativo['convergencia']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if conv['cruzamento_diario']:
                    cor = "🟢" if conv['cruzamento_diario'] == 'COMPRA' else "🔴"
                    barras = f"({conv['barra_cruzamento_diario']} barra(s))"
                    st.info(f"📅 Diário: {cor} {conv['cruzamento_diario']} {barras}")
                else:
                    st.warning("📅 Diário: Sem cruzamento")
            
            with col2:
                if conv['cruzamento_semanal']:
                    cor = "🟢" if conv['cruzamento_semanal'] == 'COMPRA' else "🔴"
                    barras = f"({conv['barra_cruzamento_semanal']} barra(s))"
                    st.info(f"📆 Semanal: {cor} {conv['cruzamento_semanal']} {barras}")
                else:
                    st.warning("📆 Semanal: Sem cruzamento")
            
            with col3:
                if conv['convergente']:
                    st.success("✅ Convergente")
                else:
                    st.error("❌ Sem convergência")
        
        # Seletor de timeframe
        timeframe = st.radio(
            "Timeframe para Visualização",
            options=["Diário", "Semanal"],
            horizontal=True
        )
        
        # Criar e mostrar gráfico
        fig = criar_grafico_cacaus_channel(
            dados_ativo['df_diario'],
            dados_ativo['df_semanal'],
            ativo_selecionado,
            timeframe
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Erro ao gerar gráfico")
    
    else:
        st.info("👈 Execute o screener na barra lateral para visualizar gráficos")
        
        # Permitir visualizar ativo individual sem screener
        st.markdown("---")
        st.subheader("🔍 Análise Individual")
        
        ticker_individual = st.text_input("Digite o ticker do ativo", value="PETR4")
        
        if st.button("📊 Visualizar Ativo", use_container_width=True):
            
            with st.spinner(f"Carregando dados de {ticker_individual}..."):
                
                try:
                    df_ativo = obter_dados_historicos_completos(ticker_individual, data_inicio, data_fim_dt)
                    
                    if not df_ativo.empty and len(df_ativo) >= max(periodo_superior, periodo_inferior, ema_periodo) + 10:
                        
                        df_diario = calcular_cacaus_channel(df_ativo, periodo_superior, periodo_inferior, ema_periodo)
                        df_semanal_raw = resample_para_semanal(df_ativo)
                        
                        if not df_semanal_raw.empty:
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
                            
                            st.success(f"✅ Dados de {ticker_individual} carregados com sucesso!")
                            st.rerun()
                        else:
                            st.error("❌ Erro ao criar timeframe semanal")
                    
                    else:
                        st.error(f"❌ Dados insuficientes para {ticker_individual}")
                
                except Exception as e:
                    st.error(f"❌ Erro ao carregar {ticker_individual}: {str(e)}")


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
    
    **REENTRADA DIÁRIO** ocorre quando o timeframe semanal já estava posicionado e o diário acabou de cruzar, oferecendo oportunidade de entrada em tendência já estabelecida no prazo maior.
    
    **REENTRADA SEMANAL** acontece quando o diário já estava posicionado e o semanal acabou de cruzar, confirmando a tendência de curto prazo com movimento de longo prazo.
    
    **RECENTE** identifica situações onde ambos os cruzamentos ocorreram há poucas barras (dentro do lookback configurado), mas não simultaneamente.
    
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
