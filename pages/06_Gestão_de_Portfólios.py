"""
Página de Gestão de Portfólios
Permite criar, editar, salvar e comparar múltiplos portfólios
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Gestão de Portfólios",
    page_icon="📁",
    layout="wide"
)

# Painel de cache na sidebar
cache_manager.exibir_painel_controle()


# ==========================================
# TÍTULO
# ==========================================

st.title("📁 Gestão de Portfólios")
st.markdown("Crie, salve e compare múltiplos portfólios de investimentos")

st.markdown("---")


# ==========================================
# TABS PRINCIPAIS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Criar/Editar",
    "💾 Meus Portfólios",
    "⚖️ Comparar",
    "📊 Análise Detalhada"
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
                    # Aqui você poderia preencher os campos automaticamente
    
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
# RODAPÉ
# ==========================================

st.markdown("---")
st.markdown("💡 **Dica:** Use a aba 'Comparar' para visualizar diferenças entre estratégias!")
