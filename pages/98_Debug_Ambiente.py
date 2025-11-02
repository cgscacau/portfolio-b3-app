"""
Debug do ambiente e dependências
"""

import streamlit as st
import sys
import pkg_resources

st.title("🔍 Debug do Ambiente")

st.header("1. Versões das Bibliotecas")

bibliotecas = [
    'yfinance',
    'pandas',
    'numpy',
    'requests',
    'streamlit',
    'python'
]

for lib in bibliotecas:
    try:
        if lib == 'python':
            st.info(f"**{lib}**: {sys.version}")
        else:
            version = pkg_resources.get_distribution(lib).version
            st.success(f"**{lib}**: {version}")
    except:
        st.error(f"**{lib}**: não instalado")

st.markdown("---")

st.header("2. Teste de Requisição Direta")

import requests

st.subheader("Teste 1: Yahoo Finance direto")

try:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/PETR4.SA"
    params = {
        'interval': '1d',
        'range': '5d'
    }
    
    response = requests.get(url, params=params, timeout=10)
    st.write(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        st.success("✓ Yahoo Finance respondeu!")
        
        with st.expander("Ver resposta"):
            st.json(data)
    else:
        st.error(f"✗ Status: {response.status_code}")
        st.code(response.text)
        
except Exception as e:
    st.error(f"✗ Erro: {str(e)}")

st.markdown("---")

st.subheader("Teste 2: User Agent")

# Testar com diferentes User Agents
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'yfinance/0.2.37',
    None
]

for idx, ua in enumerate(user_agents, 1):
    st.write(f"**Tentativa {idx}:** {ua or 'Sem User Agent'}")
    
    try:
        headers = {'User-Agent': ua} if ua else {}
        response = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/PETR4.SA",
            headers=headers,
            params={'interval': '1d', 'range': '5d'},
            timeout=10
        )
        
        if response.status_code == 200:
            st.success(f"  ✓ Funcionou! Status: {response.status_code}")
        else:
            st.warning(f"  ⚠ Status: {response.status_code}")
            
    except Exception as e:
        st.error(f"  ✗ Erro: {str(e)}")

st.markdown("---")

st.header("3. Teste yfinance Passo a Passo")

import yfinance as yf

st.subheader("Configuração do yfinance")

# Verificar se há configurações globais
st.code(f"""
yfinance.__version__ = {yf.__version__}
""")

st.subheader("Teste com diferentes métodos")

ticker_test = "PETR4.SA"

# Método 1: Ticker padrão
st.write("**Método 1: Ticker padrão**")
try:
    ticker = yf.Ticker(ticker_test)
    st.success("✓ Ticker criado")
    
    # Tentar acessar diferentes propriedades
    try:
        info = ticker.info
        st.write(f"  - info: {len(info) if info else 0} campos")
    except Exception as e:
        st.error(f"  - info: {str(e)}")
    
    try:
        hist = ticker.history(period="5d")
        st.write(f"  - history(period): {len(hist)} registros")
    except Exception as e:
        st.error(f"  - history(period): {str(e)}")
    
    try:
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=5)
        hist = ticker.history(start=start, end=end)
        st.write(f"  - history(start/end): {len(hist)} registros")
    except Exception as e:
        st.error(f"  - history(start/end): {str(e)}")
        
except Exception as e:
    st.error(f"✗ Erro ao criar ticker: {str(e)}")

# Método 2: Download direto
st.write("**Método 2: yf.download**")
try:
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=5)
    
    data = yf.download(ticker_test, start=start, end=end, progress=False)
    st.write(f"  - Registros: {len(data)}")
    st.write(f"  - Colunas: {list(data.columns)}")
    
    if not data.empty:
        st.dataframe(data.head())
    
except Exception as e:
    st.error(f"  - Erro: {str(e)}")

st.markdown("---")

st.header("4. Comparação com App que Funciona")

st.info("""
**Para comparar:**

1. Acesse um dos seus apps que funciona
2. Copie as versões das bibliotecas
3. Compare com as versões acima
4. Verifique se há diferenças no requirements.txt
""")

st.code("""
# Cole aqui as versões do app que funciona:
yfinance==?
pandas==?
requests==?
""")
