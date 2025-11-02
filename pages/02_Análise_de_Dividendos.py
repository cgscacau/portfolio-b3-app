"""
💰 Análise de Dividendos
Histórico, regularidade e calendário mensal simulado de dividendos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Configurar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data
from core.init import init_all
from core.cache import salvar_dados_cache, carregar_dados_cache, info_cache

# Configuração da página
st.set_page_config(
    page_title="Análise de Dividendos",
    page_icon="💰",
    layout="wide"
)

# Inicializar
init_all()

# ADICIONAR: Inicializar estado da análise
if 'analise_dividendos_completa' not in st.session_state:
    st.session_state.analise_dividendos_completa = False

if 'metricas_dividendos' not in st.session_state:
    st.session_state.metricas_dividendos = None

if 'calendario_dividendos' not in st.session_state:
    st.session_state.calendario_dividendos = None


# ... (todas as funções de cálculo permanecem iguais)


def carregar_dados_com_cache(tickers, data_inicio, data_fim):
    """
    Carrega dados usando cache global
    
    Args:
        tickers: Lista de tickers
        data_inicio: Data inicial
        data_fim: Data final
        
    Returns:
        Tuple (precos_df, dividendos_dict)
    """
    # Tentar carregar do cache
    price_data, dividend_data = carregar_dados_cache(tickers, data_inicio, data_fim)
    
    if price_data is not None:
        st.info("📦 Dados carregados do cache (rápido!)")
        return price_data, dividend_data if dividend_data else {}
    
    # Se não existe no cache, baixar
    st.info("📥 Baixando dados do mercado (primeira vez)...")
    
    # Carregar preços
    price_data = data.get_price_history(tickers, data_inicio, data_fim, use_cache=True)
    
    # Carregar dividendos
    dividendos_dict = {}
    
    progress_bar = st.progress(0)
    for idx, ticker in enumerate(tickers):
        try:
            divs = data.get_dividends(ticker, data_inicio, data_fim)
            if not divs.empty:
                dividendos_dict[ticker] = divs
        except:
            continue
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    
    # Salvar no cache
    salvar_dados_cache(tickers, data_inicio, data_fim, price_data, dividendos_dict)
    
    return price_data, dividendos_dict


def main():
    """Função principal da página"""
    
    st.title("💰 Análise de Dividendos")
    st.markdown("Histórico, regularidade e calendário mensal simulado de dividendos")
    st.markdown("---")
    
    # Verificar ativos selecionados
    if not st.session_state.portfolio_tickers:
        st.warning("⚠️ Nenhum ativo selecionado")
        st.info("👉 Vá para a página **Selecionar Ativos** para escolher os ativos")
        
        # Limpar análise anterior
        st.session_state.analise_dividendos_completa = False
        st.stop()
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Info do cache
        cache_info = info_cache()
        if cache_info['entries'] > 0:
            st.success(f"📦 {cache_info['entries']} conjuntos de dados em cache")
            if st.button("🗑️ Limpar Cache", help="Força novo download dos dados"):
                from core.cache import limpar_cache
                limpar_cache()
                st.session_state.analise_dividendos_completa = False
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("📅 Período de Análise")
        
        # Opções rápidas
        periodo_opcao = st.radio(
            "Período",
            ["1 ano", "2 anos", "5 anos", "Personalizado"],
            horizontal=True
        )
        
        if periodo_opcao == "Personalizado":
            col1, col2 = st.columns(2)
            with col1:
                data_inicio = st.date_input(
                    "Início",
                    value=st.session_state.period_start
                )
            with col2:
                data_fim = st.date_input(
                    "Fim",
                    value=st.session_state.period_end
                )
        else:
            anos = {"1 ano": 1, "2 anos": 2, "5 anos": 5}[periodo_opcao]
            data_fim = datetime.now()
            data_inicio = data_fim - timedelta(days=anos*365)
        
        # Atualizar session state
        novo_start = datetime.combine(data_inicio, datetime.min.time())
        novo_end = datetime.combine(data_fim, datetime.min.time())
        
        # Verificar se período mudou
        periodo_mudou = (
            novo_start != st.session_state.period_start or
            novo_end != st.session_state.period_end
        )
        
        st.session_state.period_start = novo_start
        st.session_state.period_end = novo_end
        
        if periodo_mudou:
            st.session_state.analise_dividendos_completa = False
        
        st.markdown("---")
        
        # Botão de análise
        btn_analisar = st.button(
            "📊 Analisar Dividendos",
            type="primary",
            use_container_width=True,
            help="Carrega dados e calcula métricas"
        )
    
    # Informações dos ativos
    st.info(f"📊 **{len(st.session_state.portfolio_tickers)} ativos** selecionados para análise")
    
    with st.expander("📋 Ver lista de ativos"):
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            with cols[idx % 5]:
                st.write(f"• {ticker}")
    
    st.markdown("---")
    
    # Executar análise APENAS quando botão for clicado
    if btn_analisar:
        
        # Carregar dados COM CACHE
        with st.spinner("📥 Carregando dados..."):
            try:
                precos_df, dividendos_dict = carregar_dados_com_cache(
                    st.session_state.portfolio_tickers,
                    st.session_state.period_start,
                    st.session_state.period_end
                )
                
                if precos_df.empty:
                    st.error("❌ Não foi possível carregar dados de preços")
                    st.stop()
                
                if not dividendos_dict:
                    st.warning("⚠️ Nenhum dividendo encontrado no período")
                    st.stop()
                
                st.success(f"✅ Dados carregados: **{len(dividendos_dict)} ativos** com dividendos")
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar dados: {str(e)}")
                st.stop()
        
        # Calcular métricas
        with st.spinner("🧮 Calculando métricas..."):
            metricas_lista = []
            calendario_carteira = {}
            
            for ticker, divs_df in dividendos_dict.items():
                # Preço médio do período
                if ticker in precos_df.columns:
                    preco_medio = precos_df[ticker].mean()
                else:
                    preco_medio = 0
                
                # Dividend Yield
                dy = calcular_dividend_yield(divs_df, preco_medio)
                
                # Dividendos mensais
                divs_mensais = agrupar_dividendos_por_mes(divs_df)
                
                # Regularidade
                regularidade = calcular_regularidade(divs_mensais)
                
                # Número de pagamentos
                num_pagamentos = len(divs_df)
                
                # Total de dividendos
                total_divs = divs_df['valor'].sum()
                
                metricas_lista.append({
                    'ticker': ticker,
                    'dy_anual': dy,
                    'regularidade': regularidade,
                    'num_pagamentos': num_pagamentos,
                    'total_dividendos': total_divs,
                    'preco_medio': preco_medio
                })
                
                # Calendário completo
                calendario_completo = criar_calendario_completo(
                    divs_mensais,
                    st.session_state.period_start,
                    st.session_state.period_end
                )
                calendario_carteira[ticker] = calendario_completo
            
            metricas_df = pd.DataFrame(metricas_lista)
            
            # SALVAR NO SESSION STATE
            st.session_state.metricas_dividendos = metricas_df
            st.session_state.calendario_dividendos = calendario_carteira
            st.session_state.analise_dividendos_completa = True
    
    # EXIBIR RESULTADOS SE ANÁLISE ESTIVER COMPLETA
    if st.session_state.analise_dividendos_completa:
        
        metricas_df = st.session_state.metricas_dividendos
        calendario_carteira = st.session_state.calendario_dividendos
        
        # ... (todo o código de exibição de resultados permanece igual)
        # Copie aqui toda a seção de exibição de resultados do código anterior
        
        st.header("📊 Visão Geral")
        
        # Métricas resumidas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Ativos com Dividendos",
                len(metricas_df),
                help="Número de ativos que pagaram dividendos no período"
            )
        
        with col2:
            total_pagamentos = metricas_df['num_pagamentos'].sum()
            st.metric(
                "Total de Pagamentos",
                f"{total_pagamentos}",
                help="Soma de todos os eventos de pagamento"
            )
        
        with col3:
            dy_medio = metricas_df['dy_anual'].mean()
            st.metric(
                "DY Médio",
                f"{dy_medio:.2f}%",
                help="Dividend Yield médio dos ativos"
            )
        
        with col4:
            reg_media = metricas_df['regularidade'].mean()
            st.metric(
                "Regularidade Média",
                f"{reg_media:.0f}/100",
                help="Índice médio de regularidade (0-100)"
            )
        
        # ... (resto dos gráficos e tabelas)
    
    else:
        # Mensagem inicial
        st.info("👈 Configure o período na barra lateral e clique em **Analisar Dividendos**")


if __name__ == "__main__":
    main()
