"""
Página 5: Resumo Executivo
Recomendação final, quantidades de ações e relatórios
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import io

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from core import data, metrics, opt, ui
import logging

logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Resumo Executivo - Portfolio B3",
    page_icon="📋",
    layout="wide"
)


def initialize_session_state():
    """Inicializa variáveis de sessão."""
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if 'specialized_portfolios' not in st.session_state:
        st.session_state.specialized_portfolios = {}
    
    if 'optimized_portfolios' not in st.session_state:
        st.session_state.optimized_portfolios = {}
    
    if 'recommended_portfolio' not in st.session_state:
        st.session_state.recommended_portfolio = None
    
    if 'share_quantities' not in st.session_state:
        st.session_state.share_quantities = {}


def check_prerequisites():
    """Verifica se há portfólios otimizados."""
    if not st.session_state.selected_tickers:
        ui.create_info_box(
            "⚠️ Nenhum ativo selecionado. Por favor, complete o fluxo de análise primeiro.",
            "warning"
        )
        return False
    
    all_portfolios = {
        **st.session_state.specialized_portfolios,
        **st.session_state.optimized_portfolios
    }
    
    if not all_portfolios:
        ui.create_info_box(
            "⚠️ Nenhum portfólio otimizado disponível. Por favor, otimize pelo menos um portfólio nas páginas anteriores.",
            "warning"
        )
        return False
    
    return True


def get_user_profile():
    """Interface para definir perfil do investidor."""
    
    ui.create_section_header(
        "👤 Perfil do Investidor",
        "Defina seu perfil para receber recomendação personalizada",
        "👤"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        objective = st.selectbox(
            "Objetivo Principal:",
            ["Renda Mensal", "Crescimento de Capital", "Balanceado"],
            help="Qual é sua prioridade ao investir?"
        )
    
    with col2:
        risk_tolerance = st.select_slider(
            "Tolerância a Risco:",
            options=["Muito Baixa", "Baixa", "Moderada", "Alta", "Muito Alta"],
            value="Moderada",
            help="Quanto risco você está disposto a assumir?"
        )
    
    with col3:
        time_horizon = st.selectbox(
            "Horizonte de Investimento:",
            ["Curto Prazo (< 2 anos)", "Médio Prazo (2-5 anos)", "Longo Prazo (> 5 anos)"],
            help="Por quanto tempo pretende manter o investimento?"
        )
    
    # Preferências adicionais
    with st.expander("⚙️ Preferências Adicionais", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            prefer_dividends = st.checkbox(
                "Priorizar dividendos regulares",
                value=(objective == "Renda Mensal"),
                help="Dar preferência a ativos com histórico de dividendos consistentes"
            )
        
        with col2:
            prefer_liquidity = st.checkbox(
                "Priorizar liquidez",
                value=True,
                help="Preferir ativos mais líquidos (fáceis de comprar/vender)"
            )
    
    profile = {
        'objective': objective,
        'risk_tolerance': risk_tolerance,
        'time_horizon': time_horizon,
        'prefer_dividends': prefer_dividends,
        'prefer_liquidity': prefer_liquidity
    }
    
    return profile


def recommend_portfolio(user_profile: dict):
    """Recomenda portfólio baseado no perfil do usuário."""
    
    all_portfolios = {
        **st.session_state.specialized_portfolios,
        **st.session_state.optimized_portfolios
    }
    
    if not all_portfolios:
        return None
    
    # Sistema de pontuação
    scores = {}
    
    for name, portfolio in all_portfolios.items():
        stats = portfolio['stats']
        score = 0
        
        # Pontuação por objetivo
        if user_profile['objective'] == 'Renda Mensal':
            # Priorizar dividend yield e regularidade
            if 'annual_yield' in stats:
                score += stats['annual_yield'] * 500  # Peso alto para yield
            
            # Penalizar alta volatilidade
            score -= stats['volatility'] * 100
            
            # Bonus para portfólio de dividendos
            if 'Dividendos' in name:
                score += 50
        
        elif user_profile['objective'] == 'Crescimento de Capital':
            # Priorizar retorno esperado
            score += stats['expected_return'] * 200
            
            # Sharpe também é importante
            score += stats['sharpe_ratio'] * 30
            
            # Bonus para Máximo Sharpe
            if 'Sharpe' in name:
                score += 30
        
        else:  # Balanceado
            # Priorizar Sharpe
            score += stats['sharpe_ratio'] * 50
            
            # Retorno moderado
            score += stats['expected_return'] * 100
            
            # Penalizar extremos de volatilidade
            if stats['volatility'] < 0.15:  # Muito conservador
                score -= 20
            elif stats['volatility'] > 0.30:  # Muito agressivo
                score -= 30
        
        # Ajuste por tolerância a risco
        risk_multipliers = {
            'Muito Baixa': {'volatility': -200, 'return': 50},
            'Baixa': {'volatility': -100, 'return': 75},
            'Moderada': {'volatility': -50, 'return': 100},
            'Alta': {'volatility': 0, 'return': 150},
            'Muito Alta': {'volatility': 50, 'return': 200}
        }
        
        multiplier = risk_multipliers[user_profile['risk_tolerance']]
        score += stats['volatility'] * multiplier['volatility']
        score += stats['expected_return'] * multiplier['return']
        
        # Ajuste por horizonte
        if user_profile['time_horizon'] == 'Curto Prazo (< 2 anos)':
            # Penalizar alta volatilidade
            score -= stats['volatility'] * 50
        elif user_profile['time_horizon'] == 'Longo Prazo (> 5 anos)':
            # Recompensar retorno de longo prazo
            score += stats['expected_return'] * 50
        
        # Preferências adicionais
        if user_profile.get('prefer_dividends') and 'annual_yield' in stats:
            score += stats['annual_yield'] * 200
        
        scores[name] = score
    
    # Retornar portfólio com maior score
    recommended = max(scores, key=scores.get)
    
    return recommended, scores


def calculate_share_quantities(weights: dict, total_investment: float):
    """Calcula quantidades de ações respeitando lotes."""
    
    # Obter preços atuais
    tickers = list(weights.keys())
    current_prices = data.get_current_prices(tickers)
    
    if not current_prices:
        st.error("❌ Não foi possível obter preços atuais")
        return {}, {}, 0
    
    quantities = {}
    allocated = 0
    lot_size = 100  # Lote padrão B3
    
    # Primeira passagem: calcular quantidades ideais
    for ticker, weight in weights.items():
        if ticker not in current_prices:
            continue
        
        target_value = total_investment * weight
        price = current_prices[ticker]
        
        # Quantidade ideal
        ideal_qty = target_value / price
        
        # Arredondar para lote
        lot_qty = round(ideal_qty / lot_size) * lot_size
        
        # Mínimo 1 lote
        lot_qty = max(lot_size, lot_qty)
        
        quantities[ticker] = int(lot_qty)
        allocated += quantities[ticker] * price
    
    # Segunda passagem: ajustar se exceder orçamento
    while allocated > total_investment * 1.05:  # Tolerância de 5%
        # Remover 1 lote do ativo com menor peso
        sorted_by_weight = sorted(weights.items(), key=lambda x: x[1])
        
        for ticker, _ in sorted_by_weight:
            if ticker in quantities and quantities[ticker] > lot_size:
                quantities[ticker] -= lot_size
                allocated -= lot_size * current_prices[ticker]
                break
    
    # Calcular pesos efetivos
    effective_weights = {}
    for ticker, qty in quantities.items():
        if ticker in current_prices:
            effective_weights[ticker] = (qty * current_prices[ticker]) / allocated
    
    return quantities, effective_weights, allocated


def show_recommendation():
    """Exibe recomendação de portfólio."""
    
    ui.create_section_header(
        "🎯 Recomendação Personalizada",
        "Portfólio ideal para seu perfil",
        "🎯"
    )
    
    # Obter perfil
    user_profile = get_user_profile()
    
    if st.button("🔮 Gerar Recomendação", type="primary", use_container_width=True):
        
        with st.spinner("Analisando seu perfil e gerando recomendação..."):
            
            recommended, scores = recommend_portfolio(user_profile)
            
            if not recommended:
                st.error("❌ Não foi possível gerar recomendação")
                return
            
            st.session_state.recommended_portfolio = recommended
            
            st.success(f"✅ Recomendação gerada: **{recommended}**")
            
            # Explicação da recomendação
            ui.create_info_box(
                f"Com base no seu perfil, o portfólio **{recommended}** é o mais adequado para seus objetivos.",
                "success"
            )
            
            # Detalhes da pontuação
            with st.expander("📊 Como chegamos a esta recomendação?", expanded=False):
                st.markdown("### Pontuação dos Portfólios")
                
                scores_df = pd.DataFrame({
                    'Portfólio': list(scores.keys()),
                    'Pontuação': list(scores.values())
                })
                scores_df = scores_df.sort_values('Pontuação', ascending=False)
                scores_df['Pontuação'] = scores_df['Pontuação'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(scores_df, use_container_width=True)
                
                st.markdown("""
                **Critérios considerados:**
                - Objetivo de investimento (peso alto)
                - Tolerância a risco (ajuste de volatilidade)
                - Horizonte de tempo
                - Preferências adicionais
                """)
            
            st.rerun()


def show_portfolio_details():
    """Exibe detalhes do portfólio recomendado."""
    
    if not st.session_state.recommended_portfolio:
        ui.create_info_box(
            "Gere uma recomendação usando o botão acima para ver os detalhes.",
            "info"
        )
        return
    
    recommended = st.session_state.recommended_portfolio
    
    # Buscar portfólio
    all_portfolios = {
        **st.session_state.specialized_portfolios,
        **st.session_state.optimized_portfolios
    }
    
    if recommended not in all_portfolios:
        st.error("❌ Portfólio recomendado não encontrado")
        return
    
    portfolio = all_portfolios[recommended]
    weights = portfolio['weights']
    stats = portfolio['stats']
    
    ui.create_section_header(
        f"📊 Detalhes: {recommended}",
        "Características do portfólio recomendado",
        "📊"
    )
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.create_metric_card(
            "Retorno Esperado",
            f"{stats['expected_return']*100:.2f}%",
            help_text="Anualizado",
            icon="📈"
        )
    
    with col2:
        ui.create_metric_card(
            "Volatilidade",
            f"{stats['volatility']*100:.2f}%",
            help_text="Risco anualizado",
            icon="📊"
        )
    
    with col3:
        ui.create_metric_card(
            "Sharpe Ratio",
            f"{stats['sharpe_ratio']:.3f}",
            help_text="Retorno por unidade de risco",
            icon="⭐"
        )
    
    with col4:
        ui.create_metric_card(
            "Nº de Ativos",
            f"{stats['num_assets']}",
            help_text="Diversificação",
            icon="🎯"
        )
    
    # Métricas adicionais se disponível
    if 'annual_yield' in stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ui.create_metric_card(
                "Dividend Yield",
                f"{stats['annual_yield']*100:.2f}%",
                help_text="Anual estimado",
                icon="💰"
            )
        
        with col2:
            if 'monthly_yield' in stats:
                ui.create_metric_card(
                    "Yield Mensal",
                    f"{stats['monthly_yield']*100:.2f}%",
                    help_text="Média mensal",
                    icon="📅"
                )
        
        with col3:
            if 'dividend_volatility' in stats:
                ui.create_metric_card(
                    "Regularidade Divs",
                    f"{1/stats['dividend_volatility']:.1f}" if stats['dividend_volatility'] > 0 else "N/A",
                    help_text="Quanto maior, mais regular",
                    icon="📊"
                )
    
    # Alocação
    st.markdown("### 📊 Composição do Portfólio")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = ui.plot_portfolio_weights(weights, f"Alocação - {recommended}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        weights_df = pd.DataFrame({
            'Ticker': list(weights.keys()),
            'Peso (%)': [w * 100 for w in weights.values()]
        })
        weights_df = weights_df.sort_values('Peso (%)', ascending=False)
        
        st.dataframe(weights_df, use_container_width=True, height=400)


def calculate_investment_plan():
    """Calcula plano de investimento com quantidades."""
    
    if not st.session_state.recommended_portfolio:
        return
    
    ui.create_section_header(
        "💵 Plano de Investimento",
        "Quantidades exatas de ações a comprar",
        "💵"
    )
    
    recommended = st.session_state.recommended_portfolio
    
    all_portfolios = {
        **st.session_state.specialized_portfolios,
        **st.session_state.optimized_portfolios
    }
    
    portfolio = all_portfolios[recommended]
    weights = portfolio['weights']
    
    # Valor a investir
    investment_amount = st.number_input(
        "Valor a investir (R$):",
        min_value=1000.0,
        max_value=10000000.0,
        value=st.session_state.investment_amount,
        step=1000.0,
        help="Valor total que deseja investir"
    )
    
    if st.button("🧮 Calcular Quantidades", type="primary", use_container_width=True):
        
        with st.spinner("Calculando quantidades e preços..."):
            
            quantities, effective_weights, total_allocated = calculate_share_quantities(
                weights,
                investment_amount
            )
            
            if not quantities:
                st.error("❌ Erro ao calcular quantidades")
                return
            
            st.session_state.share_quantities = quantities
            
            # Obter preços atuais
            current_prices = data.get_current_prices(list(weights.keys()))
            
            # Criar DataFrame detalhado
            plan_data = []
            
            for ticker in quantities.keys():
                qty = quantities[ticker]
                price = current_prices.get(ticker, 0)
                target_weight = weights.get(ticker, 0)
                effective_weight = effective_weights.get(ticker, 0)
                total_value = qty * price
                
                plan_data.append({
                    'Ticker': ticker,
                    'Quantidade': qty,
                    'Preço Atual (R$)': price,
                    'Valor Total (R$)': total_value,
                    'Peso Alvo (%)': target_weight * 100,
                    'Peso Efetivo (%)': effective_weight * 100,
                    'Diferença (%)': (effective_weight - target_weight) * 100
                })
            
            plan_df = pd.DataFrame(plan_data)
            plan_df = plan_df.sort_values('Valor Total (R$)', ascending=False)
            
            # Exibir resumo
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ui.create_metric_card(
                    "Valor Planejado",
                    f"R$ {investment_amount:,.2f}",
                    icon="💰"
                )
            
            with col2:
                ui.create_metric_card(
                    "Valor Alocado",
                    f"R$ {total_allocated:,.2f}",
                    icon="✅"
                )
            
            with col3:
                diff = total_allocated - investment_amount
                diff_pct = (diff / investment_amount) * 100 if investment_amount > 0 else 0
                
                ui.create_metric_card(
                    "Diferença",
                    f"R$ {diff:,.2f}",
                    delta=f"{diff_pct:+.2f}%",
                    icon="📊"
                )
            
            with col4:
                ui.create_metric_card(
                    "Nº de Ativos",
                    f"{len(quantities)}",
                    icon="🎯"
                )
            
            # Tabela detalhada
            st.markdown("### 📋 Plano Detalhado de Compra")
            
            # Formatar para exibição
            display_df = plan_df.copy()
            display_df['Preço Atual (R$)'] = display_df['Preço Atual (R$)'].apply(lambda x: f"R$ {x:.2f}")
            display_df['Valor Total (R$)'] = display_df['Valor Total (R$)'].apply(lambda x: f"R$ {x:,.2f}")
            display_df['Peso Alvo (%)'] = display_df['Peso Alvo (%)'].apply(lambda x: f"{x:.2f}%")
            display_df['Peso Efetivo (%)'] = display_df['Peso Efetivo (%)'].apply(lambda x: f"{x:.2f}%")
            display_df['Diferença (%)'] = display_df['Diferença (%)'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Avisos
            if abs(diff_pct) > 10:
                ui.create_info_box(
                    f"⚠️ A diferença entre o valor planejado e alocado é de {abs(diff_pct):.1f}%. "
                    "Isso ocorre devido ao arredondamento para lotes de 100 ações. "
                    "Considere ajustar o valor de investimento.",
                    "warning"
                )
            
            # Download
            col1, col2 = st.columns(2)
            
            with col1:
                ui.create_download_button(
                    plan_df,
                    f"plano_investimento_{recommended.replace(' ', '_')}.csv",
                    "📥 Download Plano (CSV)",
                    "csv"
                )
            
            with col2:
                # Criar texto formatado para ordem de compra
                order_text = f"PLANO DE INVESTIMENTO - {recommended}\n"
                order_text += f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
                order_text += f"Valor Total: R$ {total_allocated:,.2f}\n\n"
                order_text += "ORDENS DE COMPRA:\n"
                order_text += "-" * 50 + "\n"
                
                for _, row in plan_df.iterrows():
                    order_text += f"\n{row['Ticker']}\n"
                    order_text += f"  Quantidade: {row['Quantidade']} ações\n"
                    order_text += f"  Preço ref.: R$ {row['Preço Atual (R$)']}\n"
                    order_text += f"  Valor total: R$ {row['Valor Total (R$)']}\n"
                
                ui.create_download_button(
                    order_text,
                    f"ordens_compra_{recommended.replace(' ', '_')}.txt",
                    "📥 Download Ordens (TXT)",
                    "txt"
                )


def generate_executive_report():
    """Gera relatório executivo completo."""
    
    if not st.session_state.recommended_portfolio:
        return
    
    ui.create_section_header(
        "📄 Relatório Executivo",
        "Documento completo para download",
        "📄"
    )
    
    st.markdown("""
    O relatório executivo inclui:
    - Resumo do perfil do investidor
    - Portfólio recomendado e justificativa
    - Métricas detalhadas de risco e retorno
    - Plano de investimento com quantidades
    - Projeções de dividendos (se aplicável)
    - Avisos e disclaimers
    """)
    
    if st.button("📄 Gerar Relatório Completo", use_container_width=True):
        
        with st.spinner("Gerando relatório..."):
            
            # Criar relatório em texto
            report = generate_text_report()
            
            # Exibir preview
            with st.expander("👁️ Preview do Relatório", expanded=True):
                st.text(report)
            
            # Download
            ui.create_download_button(
                report,
                f"relatorio_executivo_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                "📥 Download Relatório",
                "txt"
            )


def generate_text_report():
    """Gera relatório em formato texto."""
    
    recommended = st.session_state.recommended_portfolio
    
    all_portfolios = {
        **st.session_state.specialized_portfolios,
        **st.session_state.optimized_portfolios
    }
    
    portfolio = all_portfolios[recommended]
    weights = portfolio['weights']
    stats = portfolio['stats']
    
    report = f"""
{'='*80}
RELATÓRIO EXECUTIVO - ANÁLISE DE PORTFÓLIO B3
{'='*80}

Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Período de Análise: {st.session_state.period_start.strftime('%d/%m/%Y')} a {st.session_state.period_end.strftime('%d/%m/%Y')}
Taxa Livre de Risco: {st.session_state.risk_free_rate*100:.2f}% a.a.

{'='*80}
1. PORTFÓLIO RECOMENDADO
{'='*80}

Estratégia: {recommended}

MÉTRICAS DE DESEMPENHO:
- Retorno Esperado (anualizado): {stats['expected_return']*100:.2f}%
- Volatilidade (anualizada): {stats['volatility']*100:.2f}%
- Índice de Sharpe: {stats['sharpe_ratio']:.3f}
- Número de Ativos: {stats['num_assets']}
- Peso Máximo Individual: {stats['max_weight']*100:.2f}%

"""
    
    if 'annual_yield' in stats:
        report += f"""
MÉTRICAS DE DIVIDENDOS:
- Dividend Yield Anual: {stats['annual_yield']*100:.2f}%
- Dividend Yield Mensal: {stats.get('monthly_yield', 0)*100:.2f}%
"""
    
    report += f"""
{'='*80}
2. COMPOSIÇÃO DO PORTFÓLIO
{'='*80}

"""
    
    weights_df = pd.DataFrame({
        'Ticker': list(weights.keys()),
        'Peso (%)': [w * 100 for w in weights.values()]
    })
    weights_df = weights_df.sort_values('Peso (%)', ascending=False)
    
    for _, row in weights_df.iterrows():
        report += f"{row['Ticker']:10s} {row['Peso (%)']:6.2f}%\n"
    
    if st.session_state.share_quantities:
        report += f"""
{'='*80}
3. PLANO DE INVESTIMENTO
{'='*80}

Valor a Investir: R$ {st.session_state.investment_amount:,.2f}

QUANTIDADES POR ATIVO:
"""
        
        current_prices = data.get_current_prices(list(weights.keys()))
        
        for ticker, qty in st.session_state.share_quantities.items():
            price = current_prices.get(ticker, 0)
            value = qty * price
            report += f"\n{ticker}\n"
            report += f"  Quantidade: {qty} ações\n"
            report += f"  Preço: R$ {price:.2f}\n"
            report += f"  Valor Total: R$ {value:,.2f}\n"
    
    report += f"""
{'='*80}
4. AVISOS E DISCLAIMERS
{'='*80}

Este relatório é gerado por uma ferramenta de análise quantitativa e não
constitui recomendação de investimento. As projeções são baseadas em dados
históricos e não garantem resultados futuros.

RISCOS:
- Volatilidade do mercado
- Risco de liquidez
- Risco de concentração setorial
- Risco cambial (se aplicável)
- Risco de crédito

RECOMENDAÇÕES:
- Consulte um profissional certificado antes de investir
- Diversifique seus investimentos
- Reavalie periodicamente sua carteira
- Mantenha reserva de emergência

{'='*80}
Relatório gerado por Portfolio B3 Analytics
© 2025 - Todos os direitos reservados
{'='*80}
"""
    
    return report


def main():
    """Função principal da página."""
    
    initialize_session_state()
    
    # Header
    st.markdown('<p class="gradient-title">📋 Resumo Executivo</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Recomendação final personalizada com plano de investimento detalhado e quantidades 
    exatas de ações a comprar.
    """)
    
    # Verificar pré-requisitos
    if not check_prerequisites():
        st.stop()
    
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Recomendação",
        "📊 Detalhes",
        "💵 Plano de Investimento",
        "📄 Relatório"
    ])
    
    with tab1:
        show_recommendation()
    
    with tab2:
        show_portfolio_details()
    
    with tab3:
        calculate_investment_plan()
    
    with tab4:
        generate_executive_report()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem 0;">
            <p>✅ Análise completa! Você pode voltar às páginas anteriores para ajustar ou explorar outras opções.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
