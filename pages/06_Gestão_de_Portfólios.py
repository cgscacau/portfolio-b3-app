"""
Página de Gestão de Portfólios
Permite criar, editar, salvar e comparar múltiplos portfólios
"""

import streamlit as st

# ==========================================
# CONFIGURAÇÃO DA PÁGINA (DEVE SER PRIMEIRO)
# ==========================================

st.set_page_config(
    page_title="Gestão de Portfólios",
    page_icon="📁",
    layout="wide"
)

# ==========================================
# IMPORTS
# ==========================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# Importar módulos
from core.portfolio import (
    portfolio_manager,
    Portfolio,
    criar_portfolio,
    salvar_portfolio,
    carregar_portfolio,
    deletar_portfolio,
    listar_portfolios,
    definir_portfolio_ativo,
    obter_portfolio_ativo
)
from core.data import get_price_history, obter_preco_atual
from core.cache import cache_manager


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def calcular_metricas_portfolio(df_precos, pesos):
    """Calcula métricas de performance do portfólio"""
    if df_precos.empty:
        return None
    
    # Retornos
    df_retornos = df_precos.pct_change().dropna()
    
    # Retorno do portfólio
    retorno_portfolio = (df_retornos * pesos).sum(axis=1)
    
    # Métricas
    retorno_acumulado = (1 + retorno_portfolio).cumprod()
    retorno_total = (retorno_acumulado.iloc[-1] - 1) * 100
    
    # Anualizar
    dias = len(retorno_portfolio)
    anos = dias / 252
    retorno_anual = ((1 + retorno_total/100) ** (1/anos) - 1) * 100 if anos > 0 else 0
    
    volatilidade = retorno_portfolio.std() * (252 ** 0.5) * 100
    sharpe = (retorno_anual / volatilidade) if volatilidade > 0 else 0
    
    # Drawdown
    cumulative = retorno_acumulado
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    return {
        'retorno_total': retorno_total,
        'retorno_anual': retorno_anual,
        'volatilidade': volatilidade,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'retorno_acumulado': retorno_acumulado,
        'drawdown': drawdown
    }


# ==========================================
# PAINEL DE CACHE
# ==========================================

# Painel de cache na sidebar
try:
    cache_manager.exibir_painel_controle()
except:
    pass


# ==========================================
# TÍTULO
# ==========================================

st.title("📁 Gestão de Portfólios")
st.markdown("Crie, salve e compare múltiplos portfólios de investimentos")

st.markdown("---")


# ==========================================
# TABS PRINCIPAIS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Criar/Editar",
    "💾 Meus Portfólios",
    "⚖️ Comparar",
    "📊 Análise Detalhada",
    "🎯 Otimizado vs Manual"
])


# ==========================================
# TAB 1: CRIAR/EDITAR PORTFÓLIO
# ==========================================

with tab1:
    st.subheader("Criar Novo Portfólio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Nome do portfólio
        nome_portfolio = st.text_input(
            "Nome do Portfólio *",
            placeholder="Ex: Conservador, Agressivo, Dividendos..."
        )
        
        # Descrição
        descricao = st.text_area(
            "Descrição (opcional)",
            placeholder="Descreva a estratégia deste portfólio..."
        )
    
    with col2:
        # Período de análise
        st.markdown("**Período de Análise**")
        
        data_fim = st.date_input(
            "Data Final",
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        data_inicio = st.date_input(
            "Data Inicial",
            value=datetime.now() - timedelta(days=365),
            max_value=data_fim
        )
    
    st.markdown("---")
    
    # Configuração de ativos
    st.markdown("### 🎯 Configuração de Ativos")
    
    # Número de ativos
    num_ativos = st.number_input(
        "Quantos ativos?",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )
    
    # Criar colunas para entrada de dados
    st.markdown("**Ativos e Pesos:**")
    
    tickers = []
    pesos = []
    
    # Criar linhas para cada ativo
    for i in range(num_ativos):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            ticker = st.text_input(
                f"Ativo {i+1}",
                key=f"ticker_{i}",
                placeholder="Ex: PETR4, VALE3...",
                label_visibility="collapsed"
            )
            tickers.append(ticker.upper().strip() if ticker else "")
        
        with col2:
            peso = st.number_input(
                f"Peso {i+1} (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / num_ativos,
                step=0.1,
                key=f"peso_{i}",
                label_visibility="collapsed"
            )
            pesos.append(peso)
        
        with col3:
            if ticker:
                # Buscar preço atual
                preco = obter_preco_atual(ticker)
                if preco:
                    st.metric("Preço", f"R$ {preco:.2f}", label_visibility="collapsed")
    
    # Validar soma dos pesos
    soma_pesos = sum(pesos)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Soma dos Pesos", f"{soma_pesos:.1f}%")
    
    with col2:
        if abs(soma_pesos - 100) < 0.01:
            st.success("✅ Pesos corretos!")
        else:
            st.error(f"❌ Soma deve ser 100% (faltam {100-soma_pesos:.1f}%)")
    
    with col3:
        # Botão de normalizar
        if st.button("⚖️ Normalizar Pesos", use_container_width=True):
            st.info("Pesos normalizados automaticamente ao criar")
    
    st.markdown("---")
    
    # Botões de ação
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        criar_btn = st.button("✅ Criar Portfólio", type="primary", use_container_width=True)
    
    with col2:
        salvar_btn = st.button("💾 Criar e Salvar", use_container_width=True)
    
    with col3:
        limpar_btn = st.button("🗑️ Limpar Campos", use_container_width=True)
    
    with col4:
        # Carregar portfólio existente
        portfolios_existentes = listar_portfolios()
        if portfolios_existentes:
            carregar_nome = st.selectbox(
                "Carregar",
                [""] + portfolios_existentes,
                label_visibility="collapsed"
            )
            if carregar_nome:
                portfolio_carregado = carregar_portfolio(carregar_nome)
                if portfolio_carregado:
                    st.success(f"✅ '{carregar_nome}' carregado!")
    
    # Ações dos botões
    if criar_btn or salvar_btn:
        # Validações
        if not nome_portfolio:
            st.error("❌ Nome do portfólio é obrigatório!")
        elif not all(tickers):
            st.error("❌ Preencha todos os tickers!")
        elif abs(soma_pesos - 100) > 0.01:
            st.error(f"❌ Soma dos pesos deve ser 100% (atual: {soma_pesos:.1f}%)")
        else:
            try:
                # Converter datas
                data_inicio_dt = datetime.combine(data_inicio, datetime.min.time())
                data_fim_dt = datetime.combine(data_fim, datetime.min.time())
                
                # Criar portfólio
                sucesso = criar_portfolio(
                    nome=nome_portfolio,
                    tickers=tickers,
                    pesos=pesos,
                    data_inicio=data_inicio_dt,
                    data_fim=data_fim_dt,
                    descricao=descricao
                )
                
                if sucesso:
                    st.success(f"✅ Portfólio '{nome_portfolio}' criado com sucesso!")
                    
                    # Se pediu para salvar
                    if salvar_btn:
                        if salvar_portfolio(nome_portfolio):
                            st.success(f"💾 Portfólio salvo em arquivo!")
                        else:
                            st.warning("⚠️ Erro ao salvar em arquivo")
                    
                    # Definir como ativo
                    definir_portfolio_ativo(nome_portfolio)
                    
                    st.balloons()
                else:
                    st.error(f"❌ Portfólio '{nome_portfolio}' já existe!")
                    
            except Exception as e:
                st.error(f"❌ Erro ao criar portfólio: {str(e)}")
    
    if limpar_btn:
        st.rerun()


# ==========================================
# TAB 2: MEUS PORTFÓLIOS
# ==========================================

with tab2:
    st.subheader("Portfólios Salvos")
    
    # Carregar todos do arquivo
    portfolio_manager.carregar_todos()
    
    portfolios = listar_portfolios()
    
    if not portfolios:
        st.info("📭 Nenhum portfólio criado ainda. Vá para a aba 'Criar/Editar' para começar!")
    else:
        st.success(f"📊 {len(portfolios)} portfólio(s) encontrado(s)")
        
        # Portfólio ativo
        portfolio_ativo = obter_portfolio_ativo()
        if portfolio_ativo:
            st.info(f"🎯 Portfólio ativo: **{portfolio_ativo.nome}**")
        
        st.markdown("---")
        
        # Listar portfólios
        for nome in portfolios:
            portfolio = carregar_portfolio(nome)
            
            if portfolio:
                with st.expander(f"📁 {nome}", expanded=(nome == st.session_state.portfolio_ativo)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Descrição:** {portfolio.descricao or 'Sem descrição'}")
                        st.markdown(f"**Período:** {portfolio.data_inicio.date()} até {portfolio.data_fim.date()}")
                        st.markdown(f"**Criado em:** {portfolio.criado_em.strftime('%d/%m/%Y %H:%M')}")
                        st.markdown(f"**Modificado em:** {portfolio.modificado_em.strftime('%d/%m/%Y %H:%M')}")
                        
                        # Mostrar ativos e pesos
                        st.markdown("**Composição:**")
                        df_composicao = pd.DataFrame({
                            'Ativo': portfolio.tickers,
                            'Peso (%)': [f"{p*100:.2f}%" for p in portfolio.pesos]
                        })
                        st.dataframe(df_composicao, use_container_width=True, hide_index=True)
                    
                    with col2:
                        # Botões de ação
                        if st.button(f"🎯 Ativar", key=f"ativar_{nome}", use_container_width=True):
                            definir_portfolio_ativo(nome)
                            st.success(f"✅ '{nome}' ativado!")
                            st.rerun()
                        
                        if st.button(f"💾 Salvar", key=f"salvar_{nome}", use_container_width=True):
                            if salvar_portfolio(nome):
                                st.success("✅ Salvo!")
                            else:
                                st.error("❌ Erro ao salvar")
                        
                        if st.button(f"🗑️ Deletar", key=f"deletar_{nome}", use_container_width=True):
                            if deletar_portfolio(nome):
                                st.success(f"✅ '{nome}' deletado!")
                                st.rerun()
                            else:
                                st.error("❌ Erro ao deletar")


# ==========================================
# TAB 3: COMPARAR PORTFÓLIOS
# ==========================================

with tab3:
    st.subheader("Comparar Portfólios")
    
    portfolios = listar_portfolios()
    
    if len(portfolios) < 2:
        st.warning("⚠️ Você precisa ter pelo menos 2 portfólios para comparar!")
    else:
        # Selecionar portfólios para comparar
        portfolios_selecionados = st.multiselect(
            "Selecione os portfólios para comparar",
            portfolios,
            default=portfolios[:2] if len(portfolios) >= 2 else portfolios
        )
        
        if len(portfolios_selecionados) < 2:
            st.info("👆 Selecione pelo menos 2 portfólios para comparar")
        else:
            st.markdown("---")
            
            # Tabela de comparação básica
            st.markdown("### 📊 Comparação Básica")
            df_comparacao = portfolio_manager.comparar(portfolios_selecionados)
            st.dataframe(df_comparacao, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Comparação de composição
            st.markdown("### 🎯 Composição dos Portfólios")
            
            cols = st.columns(len(portfolios_selecionados))
            
            for idx, nome in enumerate(portfolios_selecionados):
                portfolio = carregar_portfolio(nome)
                
                with cols[idx]:
                    st.markdown(f"**{nome}**")
                    
                    # Gráfico de pizza
                    fig = go.Figure(data=[go.Pie(
                        labels=portfolio.tickers,
                        values=[p*100 for p in portfolio.pesos],
                        hole=0.3
                    )])
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela de composição
                    df_comp = pd.DataFrame({
                        'Ativo': portfolio.tickers,
                        'Peso': [f"{p*100:.1f}%" for p in portfolio.pesos]
                    })
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Comparação de ativos únicos/comuns
            st.markdown("### 🔍 Análise de Ativos")
            
            # Coletar todos os ativos
            todos_ativos = set()
            ativos_por_portfolio = {}
            
            for nome in portfolios_selecionados:
                portfolio = carregar_portfolio(nome)
                ativos = set(portfolio.tickers)
                todos_ativos.update(ativos)
                ativos_por_portfolio[nome] = ativos
            
            # Ativos comuns
            ativos_comuns = set.intersection(*ativos_por_portfolio.values())
            
            # Ativos únicos
            ativos_unicos = {}
            for nome, ativos in ativos_por_portfolio.items():
                unicos = ativos - set.union(*[a for n, a in ativos_por_portfolio.items() if n != nome])
                if unicos:
                    ativos_unicos[nome] = unicos
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Ativos Únicos", len(todos_ativos))
            
            with col2:
                st.metric("Ativos Comuns", len(ativos_comuns))
                if ativos_comuns:
                    st.write(", ".join(sorted(ativos_comuns)))
            
            with col3:
                st.metric("Portfólios com Ativos Únicos", len(ativos_unicos))
            
            if ativos_unicos:
                st.markdown("**Ativos Únicos por Portfólio:**")
                for nome, unicos in ativos_unicos.items():
                    st.write(f"- **{nome}:** {', '.join(sorted(unicos))}")


# ==========================================
# TAB 4: ANÁLISE DETALHADA
# ==========================================

with tab4:
    st.subheader("Análise Detalhada de Portfólio")
    
    portfolios = listar_portfolios()
    
    if not portfolios:
        st.warning("⚠️ Nenhum portfólio disponível para análise")
    else:
        # Selecionar portfólio
        portfolio_selecionado = st.selectbox(
            "Selecione um portfólio para análise detalhada",
            portfolios,
            index=portfolios.index(st.session_state.portfolio_ativo) if st.session_state.portfolio_ativo in portfolios else 0
        )
        
        if portfolio_selecionado:
            portfolio = carregar_portfolio(portfolio_selecionado)
            
            st.markdown("---")
            
            # Informações básicas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Ativos", len(portfolio.tickers))
            
            with col2:
                dias = (portfolio.data_fim - portfolio.data_inicio).days
                st.metric("Período", f"{dias} dias")
            
            with col3:
                st.metric("Criado em", portfolio.criado_em.strftime('%d/%m/%Y'))
            
            with col4:
                st.metric("Modificado em", portfolio.modificado_em.strftime('%d/%m/%Y'))
            
            st.markdown("---")
            
            # Buscar dados históricos
            with st.spinner("Buscando dados históricos..."):
                df_precos = get_price_history(
                    portfolio.tickers,
                    portfolio.data_inicio,
                    portfolio.data_fim
                )
            
            if df_precos.empty:
                st.error("❌ Não foi possível obter dados históricos")
            else:
                # Calcular retornos
                df_retornos = df_precos.pct_change().dropna()
                
                # Retorno do portfólio
                retorno_portfolio = (df_retornos * portfolio.pesos).sum(axis=1)
                retorno_acumulado = (1 + retorno_portfolio).cumprod()
                
                # Métricas de performance
                st.markdown("### 📈 Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                retorno_total = (retorno_acumulado.iloc[-1] - 1) * 100
                retorno_anual = ((1 + retorno_total/100) ** (252/len(retorno_portfolio)) - 1) * 100
                volatilidade = retorno_portfolio.std() * (252 ** 0.5) * 100
                sharpe = (retorno_anual / volatilidade) if volatilidade > 0 else 0
                
                with col1:
                    st.metric("Retorno Total", f"{retorno_total:.2f}%")
                
                with col2:
                    st.metric("Retorno Anualizado", f"{retorno_anual:.2f}%")
                
                with col3:
                    st.metric("Volatilidade Anual", f"{volatilidade:.2f}%")
                
                with col4:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                st.markdown("---")
                
                # Gráfico de evolução
                st.markdown("### 📊 Evolução do Portfólio")
                
                fig = go.Figure()
                
                # Linha do portfólio
                fig.add_trace(go.Scatter(
                    x=retorno_acumulado.index,
                    y=retorno_acumulado.values,
                    mode='lines',
                    name='Portfólio',
                    line=dict(color='blue', width=2)
                ))
                
                # Linhas dos ativos individuais
                for ticker in portfolio.tickers:
                    if ticker in df_precos.columns:
                        retorno_ativo = (df_precos[ticker] / df_precos[ticker].iloc[0])
                        fig.add_trace(go.Scatter(
                            x=retorno_ativo.index,
                            y=retorno_ativo.values,
                            mode='lines',
                            name=ticker,
                            line=dict(width=1),
                            opacity=0.5
                        ))
                
                fig.update_layout(
                    title="Evolução do Portfólio vs Ativos Individuais",
                    xaxis_title="Data",
                    yaxis_title="Valor Acumulado (Base 1.0)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Tabela de contribuição
                st.markdown("### 🎯 Contribuição por Ativo")
                
                contribuicoes = []
                for i, ticker in enumerate(portfolio.tickers):
                    if ticker in df_retornos.columns:
                        ret_ativo = df_retornos[ticker].mean() * 252 * 100
                        contrib = ret_ativo * portfolio.pesos[i]
                        vol_ativo = df_retornos[ticker].std() * (252 ** 0.5) * 100
                        
                        contribuicoes.append({
                            'Ativo': ticker,
                            'Peso': f"{portfolio.pesos[i]*100:.2f}%",
                            'Retorno Anual': f"{ret_ativo:.2f}%",
                            'Contribuição': f"{contrib:.2f}%",
                            'Volatilidade': f"{vol_ativo:.2f}%"
                        })
                
                df_contrib = pd.DataFrame(contribuicoes)
                st.dataframe(df_contrib, use_container_width=True, hide_index=True)


# ==========================================
# TAB 5: OTIMIZADO VS MANUAL
# ==========================================

with tab5:
    st.subheader("🎯 Portfólio Otimizado vs Manual")
    st.markdown("Compare seu portfólio manual com o portfólio otimizado pelo sistema")
    
    portfolios = listar_portfolios()
    
    if not portfolios:
        st.warning("⚠️ Nenhum portfólio disponível. Crie um portfólio na aba 'Criar/Editar' primeiro!")
    else:
        # Seleção do portfólio manual
        st.markdown("### 📁 Selecione seu Portfólio Manual")
        
        portfolio_manual_nome = st.selectbox(
            "Portfólio Manual",
            portfolios,
            index=portfolios.index(st.session_state.portfolio_ativo) if st.session_state.portfolio_ativo in portfolios else 0
        )
        
        portfolio_manual = carregar_portfolio(portfolio_manual_nome)
        
        if not portfolio_manual:
            st.error("❌ Erro ao carregar portfólio")
        else:
            st.markdown("---")
            
            # Configurações de otimização
            st.markdown("### ⚙️ Configurações de Otimização")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metodo_otimizacao = st.selectbox(
                    "Método de Otimização",
                    ["Sharpe Máximo", "Mínima Volatilidade", "Retorno Máximo"],
                    help="Escolha o critério de otimização"
                )
            
            with col2:
                usar_mesmos_ativos = st.checkbox(
                    "Usar mesmos ativos do portfólio manual",
                    value=True,
                    help="Se marcado, otimiza apenas redistribuindo os pesos."
                )
            
            # Botão de otimizar
            if st.button("🚀 Calcular Portfólio Otimizado", type="primary", use_container_width=True):
                
                with st.spinner("🔄 Calculando portfólio otimizado..."):
                    
                    # Buscar dados históricos
                    df_precos = get_price_history(
                        portfolio_manual.tickers,
                        portfolio_manual.data_inicio,
                        portfolio_manual.data_fim
                    )
                    
                    if df_precos.empty:
                        st.error("❌ Não foi possível obter dados históricos")
                    else:
                        # Calcular retornos e covariância
                        df_retornos = df_precos.pct_change().dropna()
                        retornos_medios = df_retornos.mean()
                        matriz_cov = df_retornos.cov()
                        
                        # Otimização
                        num_ativos = len(portfolio_manual.tickers)
                        
                        def portfolio_stats(pesos, retornos, cov_matrix):
                            """Calcula estatísticas do portfólio"""
                            retorno = np.dot(pesos, retornos) * 252
                            volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix * 252, pesos)))
                            sharpe = retorno / volatilidade if volatilidade > 0 else 0
                            return retorno, volatilidade, sharpe
                        
                        def objetivo_sharpe(pesos, retornos, cov_matrix):
                            """Objetivo: maximizar Sharpe (minimizar -Sharpe)"""
                            _, _, sharpe = portfolio_stats(pesos, retornos, cov_matrix)
                            return -sharpe
                        
                        def objetivo_volatilidade(pesos, retornos, cov_matrix):
                            """Objetivo: minimizar volatilidade"""
                            _, volatilidade, _ = portfolio_stats(pesos, retornos, cov_matrix)
                            return volatilidade
                        
                        def objetivo_retorno(pesos, retornos, cov_matrix):
                            """Objetivo: maximizar retorno (minimizar -retorno)"""
                            retorno, _, _ = portfolio_stats(pesos, retornos, cov_matrix)
                            return -retorno
                        
                        # Restrições e bounds
                        restricoes = [
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                        ]
                        bounds = tuple((0, 1) for _ in range(num_ativos))
                        
                        # Pesos iniciais (iguais)
                        pesos_iniciais = np.array([1.0 / num_ativos] * num_ativos)
                        
                        # Escolher função objetivo
                        if metodo_otimizacao == "Sharpe Máximo":
                            objetivo = objetivo_sharpe
                        elif metodo_otimizacao == "Mínima Volatilidade":
                            objetivo = objetivo_volatilidade
                        else:
                            objetivo = objetivo_retorno
                        
                        # Otimizar
                        resultado = minimize(
                            objetivo,
                            pesos_iniciais,
                            args=(retornos_medios, matriz_cov),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=restricoes
                        )
                        
                        if not resultado.success:
                            st.error("❌ Falha na otimização")
                        else:
                            # Pesos otimizados
                            pesos_otimizados = resultado.x
                            
                            # Salvar no session_state
                            st.session_state.portfolio_otimizado = {
                                'tickers': portfolio_manual.tickers,
                                'pesos': pesos_otimizados.tolist(),
                                'metodo': metodo_otimizacao,
                                'data_calculo': datetime.now()
                            }
                            
                            st.success("✅ Portfólio otimizado calculado com sucesso!")
                            st.rerun()
            
            # Mostrar comparação se já foi calculado
            if 'portfolio_otimizado' in st.session_state:
                
                st.markdown("---")
                st.markdown("## 📊 Comparação Detalhada")
                
                # Dados do portfólio otimizado
                pesos_otimizados = st.session_state.portfolio_otimizado['pesos']
                metodo = st.session_state.portfolio_otimizado['metodo']
                
                # Buscar dados
                df_precos = get_price_history(
                    portfolio_manual.tickers,
                    portfolio_manual.data_inicio,
                    portfolio_manual.data_fim
                )
                
                if not df_precos.empty:
                    
                    # Calcular métricas para ambos
                    metricas_manual = calcular_metricas_portfolio(df_precos, portfolio_manual.pesos)
                    metricas_otimizado = calcular_metricas_portfolio(df_precos, pesos_otimizados)
                    
                    if metricas_manual and metricas_otimizado:
                        
                        # SEÇÃO 1: Comparação de Métricas
                        st.markdown("### 📈 Comparação de Performance")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # Retorno
                        with col1:
                            st.markdown("**Retorno Anualizado**")
                            
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric("Manual", f"{metricas_manual['retorno_anual']:.2f}%")
                            with subcol2:
                                delta = metricas_otimizado['retorno_anual'] - metricas_manual['retorno_anual']
                                st.metric("Otimizado", f"{metricas_otimizado['retorno_anual']:.2f}%", delta=f"{delta:+.2f}%")
                        
                        # Volatilidade
                        with col2:
                            st.markdown("**Volatilidade Anual**")
                            
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric("Manual", f"{metricas_manual['volatilidade']:.2f}%")
                            with subcol2:
                                delta = metricas_otimizado['volatilidade'] - metricas_manual['volatilidade']
                                st.metric("Otimizado", f"{metricas_otimizado['volatilidade']:.2f}%", delta=f"{delta:+.2f}%", delta_color="inverse")
                        
                        # Sharpe
                        with col3:
                            st.markdown("**Sharpe Ratio**")
                            
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric("Manual", f"{metricas_manual['sharpe']:.2f}")
                            with subcol2:
                                delta = metricas_otimizado['sharpe'] - metricas_manual['sharpe']
                                st.metric("Otimizado", f"{metricas_otimizado['sharpe']:.2f}", delta=f"{delta:+.2f}")
                        
                        st.markdown("---")
                        
                        # SEÇÃO 2: Comparação de Composição
                        st.markdown("### 🎯 Comparação de Composição")
                        
                        col1, col2 = st.columns(2)
                        
                        # Portfólio Manual
                        with col1:
                            st.markdown("**Portfólio Manual**")
                            
                            fig_manual = go.Figure(data=[go.Pie(
                                labels=portfolio_manual.tickers,
                                values=[p*100 for p in portfolio_manual.pesos],
                                hole=0.4
                            )])
                            
                            fig_manual.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                            st.plotly_chart(fig_manual, use_container_width=True)
                            
                            df_manual = pd.DataFrame({
                                'Ativo': portfolio_manual.tickers,
                                'Peso': [f"{p*100:.2f}%" for p in portfolio_manual.pesos]
                            })
                            st.dataframe(df_manual, use_container_width=True, hide_index=True)
                        
                        # Portfólio Otimizado
                        with col2:
                            st.markdown(f"**Portfólio Otimizado ({metodo})**")
                            
                            fig_otimizado = go.Figure(data=[go.Pie(
                                labels=portfolio_manual.tickers,
                                values=[p*100 for p in pesos_otimizados],
                                hole=0.4
                            )])
                            
                            fig_otimizado.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                            st.plotly_chart(fig_otimizado, use_container_width=True)
                            
                            df_otimizado = pd.DataFrame({
                                'Ativo': portfolio_manual.tickers,
                                'Peso': [f"{p*100:.2f}%" for p in pesos_otimizados]
                            })
                            st.dataframe(df_otimizado, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        
                        # SEÇÃO 3: Diferença de Pesos
                        st.markdown("### 📊 Análise de Diferenças")
                        
                        diferencas = []
                        for i, ticker in enumerate(portfolio_manual.tickers):
                            diff = (pesos_otimizados[i] - portfolio_manual.pesos[i]) * 100
                            diferencas.append({
                                'Ativo': ticker,
                                'Manual': f"{portfolio_manual.pesos[i]*100:.2f}%",
                                'Otimizado': f"{pesos_otimizados[i]*100:.2f}%",
                                'Diferença': f"{diff:+.2f}%",
                                'Ação': '📈 Aumentar' if diff > 0.5 else ('📉 Reduzir' if diff < -0.5 else '✅ Manter')
                            })
                        
                        df_diferencas = pd.DataFrame(diferencas)
                        st.dataframe(df_diferencas, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        
                        # SEÇÃO 4: Evolução Comparada
                        st.markdown("### 📈 Evolução Comparada")
                        
                        fig_evolucao = go.Figure()
                        
                        fig_evolucao.add_trace(go.Scatter(
                            x=metricas_manual['retorno_acumulado'].index,
                            y=metricas_manual['retorno_acumulado'].values,
                            mode='lines',
                            name='Manual',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_evolucao.add_trace(go.Scatter(
                            x=metricas_otimizado['retorno_acumulado'].index,
                            y=metricas_otimizado['retorno_acumulado'].values,
                            mode='lines',
                            name=f'Otimizado ({metodo})',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_evolucao.update_layout(
                            title="Evolução: Manual vs Otimizado",
                            xaxis_title="Data",
                            yaxis_title="Retorno Acumulado",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_evolucao, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # SEÇÃO 5: Recomendações
                        st.markdown("### 💡 Recomendações")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Análise Quantitativa")
                            
                            if metricas_otimizado['sharpe'] > metricas_manual['sharpe']:
                                st.success(f"✅ O portfólio otimizado tem **melhor relação risco/retorno**")
                            else:
                                st.info(f"ℹ️ O portfólio manual tem melhor Sharpe")
                            
                            if metricas_otimizado['volatilidade'] < metricas_manual['volatilidade']:
                                reducao = ((metricas_manual['volatilidade'] - metricas_otimizado['volatilidade']) / metricas_manual['volatilidade']) * 100
                                st.success(f"✅ O portfólio otimizado tem **{reducao:.1f}% menos risco**")
                            
                            if metricas_otimizado['retorno_anual'] > metricas_manual['retorno_anual']:
                                ganho = metricas_otimizado['retorno_anual'] - metricas_manual['retorno_anual']
                                st.success(f"✅ O portfólio otimizado teria gerado **{ganho:.2f}% a mais**")
                        
                        with col2:
                            st.markdown("#### 🎯 Ações Sugeridas")
                            
                            mudancas_significativas = [d for d in diferencas if abs(float(d['Diferença'].replace('%', '').replace('+', ''))) > 5]
                            
                            if mudancas_significativas:
                                st.markdown("**Ajustes recomendados:**")
                                for mudanca in mudancas_significativas:
                                    st.write(f"• {mudanca['Ação']} **{mudanca['Ativo']}** ({mudanca['Diferença']})")
                            else:
                                st.success("✅ Seu portfólio manual já está bem balanceado!")
                            
                            if st.button("💾 Salvar Portfólio Otimizado", use_container_width=True):
                                nome_otimizado = f"{portfolio_manual.nome}_Otimizado_{metodo.replace(' ', '_')}"
                                
                                sucesso = criar_portfolio(
                                    nome=nome_otimizado,
                                    tickers=portfolio_manual.tickers,
                                    pesos=pesos_otimizados,
                                    data_inicio=portfolio_manual.data_inicio,
                                    data_fim=portfolio_manual.data_fim,
                                    descricao=f"Versão otimizada de '{portfolio_manual.nome}' usando {metodo}"
                                )
                                
                                if sucesso:
                                    salvar_portfolio(nome_otimizado)
                                    st.success(f"✅ Portfólio '{nome_otimizado}' salvo!")
                                else:
                                    st.error("❌ Erro ao salvar (pode já existir)")


# ==========================================
# RODAPÉ
# ==========================================

st.markdown("---")
st.markdown("💡 **Dica:** Use a aba 'Comparar' para visualizar diferenças entre estratégias!")
