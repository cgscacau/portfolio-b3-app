"""
Página de teste para diagnosticar o download de dados
"""

import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="Teste de API", page_icon="🔧")

st.title("🔧 Diagnóstico de Download de Dados")
st.markdown("---")

# ==========================================
# TESTE 1: yfinance instalado?
# ==========================================
st.header("1️⃣ Verificar yfinance")

try:
    import yfinance
    st.success(f"✓ yfinance instalado - versão: {yfinance.__version__}")
except Exception as e:
    st.error(f"✗ Erro ao importar yfinance: {e}")
    st.stop()

st.markdown("---")

# ==========================================
# TESTE 2: Criar Ticker
# ==========================================
st.header("2️⃣ Criar Ticker")

with st.spinner("Criando ticker PETR4.SA..."):
    try:
        ticker = yf.Ticker('PETR4.SA')
        st.success("✓ Ticker criado com sucesso")
    except Exception as e:
        st.error(f"✗ Erro ao criar ticker: {e}")
        st.stop()

st.markdown("---")

# ==========================================
# TESTE 3: Obter Info
# ==========================================
st.header("3️⃣ Obter Info")

with st.spinner("Buscando informações..."):
    try:
        info = ticker.info
        if info:
            st.success(f"✓ Info obtido: {len(info)} campos")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nome", info.get('longName', 'N/A'))
            with col2:
                preco = info.get('currentPrice') or info.get('regularMarketPrice')
                if preco:
                    st.metric("Preço", f"R$ {preco:.2f}")
                else:
                    st.warning("Preço não disponível no info")
            
            with st.expander("Ver todos os campos"):
                st.json(info)
        else:
            st.warning("⚠ Info vazio")
    except Exception as e:
        st.error(f"✗ Erro ao obter info: {e}")

st.markdown("---")

# ==========================================
# TESTE 4: History com period
# ==========================================
st.header("4️⃣ History com period='5d'")

with st.spinner("Buscando histórico (5 dias)..."):
    try:
        hist = ticker.history(period='5d')
        
        if not hist.empty:
            st.success(f"✓ Histórico obtido: {len(hist)} registros")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Registros", len(hist))
            with col2:
                ultimo_preco = hist['Close'].iloc[-1]
                st.metric("Último Preço", f"R$ {ultimo_preco:.2f}")
            
            st.dataframe(hist.tail())
            st.line_chart(hist['Close'])
        else:
            st.error("✗ Histórico vazio")
            
    except Exception as e:
        st.error(f"✗ Erro ao obter histórico: {e}")
        st.code(str(e))

st.markdown("---")

# ==========================================
# TESTE 5: History com datas
# ==========================================
st.header("5️⃣ History com start/end")

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

st.info(f"Período: {start_date.date()} até {end_date.date()}")

with st.spinner("Buscando histórico (30 dias)..."):
    try:
        hist = ticker.history(start=start_date, end=end_date)
        
        if not hist.empty:
            st.success(f"✓ Histórico obtido: {len(hist)} registros")
            st.dataframe(hist.head())
            st.line_chart(hist['Close'])
        else:
            st.error("✗ Histórico vazio")
            
    except Exception as e:
        st.error(f"✗ Erro ao obter histórico: {e}")
        st.code(str(e))

st.markdown("---")

# ==========================================
# TESTE 6: yf.download - ÚNICO TICKER
# ==========================================
st.header("6️⃣ yf.download - Único Ticker")

with st.spinner("Testando yf.download('PETR4.SA')..."):
    try:
        data = yf.download(
            'PETR4.SA',
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if not data.empty:
            st.success(f"✓ Download concluído: {len(data)} registros")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Linhas", len(data))
            with col2:
                st.metric("Colunas", len(data.columns))
            
            st.write("**Colunas:**", data.columns.tolist())
            st.dataframe(data.head())
            
            if 'Close' in data.columns:
                st.line_chart(data['Close'])
        else:
            st.error("✗ Download retornou vazio")
            
    except Exception as e:
        st.error(f"✗ Erro no download: {e}")
        st.code(str(e))

st.markdown("---")

# ==========================================
# TESTE 7: yf.download - MÚLTIPLOS TICKERS
# ==========================================
st.header("7️⃣ yf.download - Múltiplos Tickers")

tickers_teste = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
st.info(f"Testando: {', '.join(tickers_teste)}")

with st.spinner("Baixando múltiplos tickers..."):
    try:
        data = yf.download(
            tickers_teste,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='column'
        )
        
        if not data.empty:
            st.success(f"✓ Download concluído")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Linhas", len(data))
            with col2:
                st.metric("Colunas", len(data.columns))
            with col3:
                st.metric("Níveis", data.columns.nlevels)
            
            st.write("**Estrutura das colunas:**")
            st.write(f"- Tipo: {type(data.columns)}")
            st.write(f"- Níveis: {data.columns.nlevels}")
            
            if isinstance(data.columns, pd.MultiIndex):
                st.write(f"- Nível 0: {data.columns.get_level_values(0).unique().tolist()}")
                st.write(f"- Nível 1: {data.columns.get_level_values(1).unique().tolist()}")
            else:
                st.write(f"- Colunas: {data.columns.tolist()}")
            
            st.dataframe(data.head())
            
            # Tentar extrair Close
            if 'Close' in data.columns:
                st.write("**Preços de Fechamento:**")
                st.line_chart(data['Close'])
            
        else:
            st.error("✗ Download retornou vazio")
            
    except Exception as e:
        st.error(f"✗ Erro no download: {e}")
        st.code(str(e))

st.markdown("---")

# ==========================================
# TESTE 8: Download sequencial
# ==========================================
st.header("8️⃣ Download Sequencial (um por vez)")

tickers_seq = ['PETR4', 'VALE3', 'ITUB4']

with st.spinner("Baixando sequencialmente..."):
    resultados = {}
    
    for ticker in tickers_seq:
        try:
            ticker_sa = f"{ticker}.SA"
            data = yf.download(
                ticker_sa,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if not data.empty and 'Close' in data.columns:
                resultados[ticker] = data['Close']
                st.success(f"✓ {ticker}: {len(data)} registros")
            else:
                st.warning(f"⚠ {ticker}: sem dados")
                
        except Exception as e:
            st.error(f"✗ {ticker}: {str(e)}")
    
    if resultados:
        df_final = pd.DataFrame(resultados)
        
        st.success(f"✓ DataFrame final criado: {df_final.shape}")
        st.dataframe(df_final.head())
        st.line_chart(df_final)
    else:
        st.error("✗ Nenhum dado obtido")

st.markdown("---")

# ==========================================
# RESUMO
# ==========================================
st.header("📊 Resumo do Diagnóstico")

st.info("""
**O que verificamos:**

1. ✅ yfinance instalado e versão
2. ✅ Criação de Ticker
3. ✅ Obtenção de info
4. ✅ History com period
5. ✅ History com start/end
6. ✅ yf.download único ticker
7. ✅ yf.download múltiplos tickers
8. ✅ Download sequencial

**Próximos passos:**
- Identifique qual método funcionou
- Verifique a estrutura dos dados retornados
- Copie a saída e me envie para ajustar o código
""")
