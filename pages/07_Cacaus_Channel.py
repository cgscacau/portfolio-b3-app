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
            'barra_cruzamento_semanal': None
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


# ==========================================
# VISUALIZAÇÃO
# ==========================================

def criar_grafico_cacaus_channel(df_diario, df_semanal, ticker, timeframe_ativo="Diário"):
    """Cria gráfico do Cacau's Channel"""
    
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
    periodo_analise = st.selectbox("Período de Análise", options=["3 meses", "6 meses", "1 ano", "2 anos"], index=2)
    
    periodos_dias = {"3 meses": 90, "6 meses": 180, "1 ano": 365, "2 anos": 730}
    
    data_inicio = datetime.combine(data_fim, datetime.min.time()) - timedelta(days=periodos_dias[periodo_analise])
    data_fim_dt = datetime.combine(data_fim, datetime.min.time())


# ==========================================
# LAYOUT EM DUAS COLUNAS
# ==========================================

col_esquerda, col_direita = st.columns([1, 2])


# ==========================================
# COLUNA ESQUERDA: SELEÇÃO E SCREENER
# ==========================================

with col_esquerda:
    
    st.subheader("📈 Seleção de Ativos")
    
    # Carregar base completa
    base_completa = carregar_base_ativos()
    
    if base_completa:
        st.success(f"✅ {len(base_completa)} ativos disponíveis")
    
    # Opções de seleção
    opcao_selecao = st.radio(
        "Fonte",
        options=[
            "📁 Portfólio",
            "🌐 Base B3",
            "✍️ Manual"
        ],
        label_visibility="collapsed"
    )
    
    tickers = []
    
    # OPÇÃO 1: Portfólio
    if opcao_selecao == "📁 Portfólio":
        try:
            from core.portfolio import listar_portfolios, carregar_portfolio
            portfolios_disponiveis = listar_portfolios()
            
            if portfolios_disponiveis:
                portfolio_selecionado = st.selectbox("Portfólio", portfolios_disponiveis, label_visibility="collapsed")
                portfolio = carregar_portfolio(portfolio_selecionado)
                tickers = portfolio.tickers if portfolio else []
                st.info(f"📊 {len(tickers)} ativos")
            else:
                st.warning("Nenhum portfólio salvo")
        except:
            st.error("Erro ao carregar portfólios")
    
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
                    tickers_filtrados.extend([t for t in base_completa if t[-1] in ['3', '4']])
                
                if "FIIs" in filtro_tipo:
                    tickers_filtrados.extend([t for t in base_completa if t.endswith('11')])
                
                if "ETFs" in filtro_tipo:
                    tickers_filtrados.extend([t for t in base_completa if 'B' in t[-2:]])
                
                tickers = sorted(list(set(tickers_filtrados)))
            
            if limite_ativos > 0 and len(tickers) > limite_ativos:
                tickers = tickers[:limite_ativos]
            
            st.info(f"📊 {len(tickers)} ativos")
    
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
        
        st.info(f"📊 {len(tickers)} ativos")
    
    # Botão de screener
    st.markdown("---")
    
    if st.button("🔍 Executar Screener", type="primary", use_container_width=True):
        
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
                
                convergencia = detectar_convergencia_com_cruzamento(df_diario, df_semanal, lookback_cruzamento)
                
                # Salvar TODOS os dados (mesmo sem convergência)
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Analisados", total_analisados)
        
        with col2:
            st.metric("Com Dados", total_com_dados)
        
        with col3:
            st.metric("🎯 Sinais", total_convergentes)
        
        if oportunidades:
            st.success(f"✅ {len(oportunidades)} oportunidade(s)!")
        else:
            st.info("ℹ️ Nenhum sinal no momento")
    
    # Mostrar screener se houver oportunidades
    if 'cacaus_oportunidades' in st.session_state and st.session_state.cacaus_oportunidades:
        
        st.markdown("---")
        st.subheader("📊 Oportunidades")
        
        oportunidades = st.session_state.cacaus_oportunidades
        
        for opp in oportunidades:
            direcao_cor = "🟢" if opp['direcao'] == 'COMPRA' else "🔴"
            
            if st.button(
                f"{direcao_cor} {opp['ticker']} - {opp['direcao']}",
                key=f"btn_{opp['ticker']}",
                use_container_width=True
            ):
                st.session_state.ativo_visualizar = opp['ticker']
                st.rerun()


# ==========================================
# COLUNA DIREITA: GRÁFICO
# ==========================================

with col_direita:
    
    st.subheader("📈 Visualização do Indicador")
    
    # Seleção de ativo para visualizar
    if 'cacaus_todos_dados' in st.session_state and st.session_state.cacaus_todos_dados:
        
        ativos_disponiveis = sorted(list(st.session_state.cacaus_todos_dados.keys()))
        
        # Usar ativo do session_state ou primeiro da lista
        ativo_padrao = st.session_state.get('ativo_visualizar', ativos_disponiveis[0])
        
        if ativo_padrao not in ativos_disponiveis:
            ativo_padrao = ativos_disponiveis[0]
        
        ativo_selecionado = st.selectbox(
            "Ativo para visualizar",
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
            # TEM SINAL - Mostrar setup completo
            st.success(f"🎯 SINAL DETECTADO: {opp_selecionada['direcao']}")
            
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
            # SEM SINAL - Apenas mostrar status
            conv = dados_ativo['convergencia']
            
            if conv['cruzamento_diario'] or conv['cruzamento_semanal']:
                st.info("ℹ️ Cruzamento detectado, mas sem convergência entre timeframes")
                
                col1, col2 = st.columns(2)
                with col1:
                    if conv['cruzamento_diario']:
                        st.write(f"📅 Diário: {conv['cruzamento_diario']}")
                with col2:
                    if conv['cruzamento_semanal']:
                        st.write(f"📆 Semanal: {conv['cruzamento_semanal']}")
            else:
                st.warning("⚠️ Nenhum cruzamento recente detectado")
        
        # Timeframe
        timeframe = st.radio(
            "Timeframe",
            options=["Diário", "Semanal"],
            horizontal=True
        )
        
        # Gráfico
        fig = criar_grafico_cacaus_channel(
            dados_ativo['df_diario'],
            dados_ativo['df_semanal'],
            ativo_selecionado,
            timeframe
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Execute o screener para visualizar os gráficos")


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")
st.markdown("""
### 📖 Lógica de Sinais (Cruzamentos)

**Sinal de COMPRA:**
- ✅ Linha Branca cruza para CIMA da Linha Laranja no **Semanal**
- ✅ Linha Branca cruza para CIMA da Linha Laranja no **Diário**
- ✅ Convergência: Ambos cruzamentos na mesma direção

**Sinal de VENDA:**
- ✅ Linha Branca cruza para BAIXO da Linha Laranja no **Semanal**
- ✅ Linha Branca cruza para BAIXO da Linha Laranja no **Diário**
- ✅ Convergência: Ambos cruzamentos na mesma direção

**Tipos de Sinal:**
- 🎯 **SIMULTÂNEO:** Cruzamento na última barra de ambos timeframes
- 📅 **REENTRADA DIÁRIO:** Semanal já estava posicionado, diário cruzou agora
- 📆 **REENTRADA SEMANAL:** Diário já estava posicionado, semanal cruzou agora
- ⏰ **RECENTE:** Ambos cruzaram nas últimas barras (lookback)

⚠️ **Aviso:** Ferramenta de análise técnica. Não constitui recomendação de investimento.
""")
