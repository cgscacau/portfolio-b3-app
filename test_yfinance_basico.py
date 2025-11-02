"""
Teste básico para verificar se yfinance está funcionando
"""

import yfinance as yf
from datetime import datetime, timedelta

print("\n" + "="*60)
print("TESTE BÁSICO DO YFINANCE")
print("="*60)

# Teste 1: Ticker simples
print("\n1. Testando yf.Ticker('PETR4.SA')...")
try:
    ticker = yf.Ticker('PETR4.SA')
    print("   ✓ Ticker criado")
except Exception as e:
    print(f"   ✗ Erro: {e}")

# Teste 2: Info
print("\n2. Testando .info...")
try:
    info = ticker.info
    print(f"   ✓ Info obtido: {len(info)} campos")
    print(f"   Nome: {info.get('longName', 'N/A')}")
except Exception as e:
    print(f"   ✗ Erro: {e}")

# Teste 3: History com period
print("\n3. Testando .history(period='5d')...")
try:
    hist = ticker.history(period='5d')
    print(f"   ✓ História obtida: {len(hist)} registros")
    if not hist.empty:
        print(f"   Último preço: R$ {hist['Close'].iloc[-1]:.2f}")
        print(hist.tail())
except Exception as e:
    print(f"   ✗ Erro: {e}")

# Teste 4: History com datas
print("\n4. Testando .history(start=..., end=...)...")
try:
    end = datetime.now()
    start = end - timedelta(days=30)
    hist = ticker.history(start=start, end=end)
    print(f"   ✓ História obtida: {len(hist)} registros")
    if not hist.empty:
        print(hist.head())
except Exception as e:
    print(f"   ✗ Erro: {e}")

# Teste 5: yf.download
print("\n5. Testando yf.download()...")
try:
    end = datetime.now()
    start = end - timedelta(days=30)
    data = yf.download('PETR4.SA', start=start, end=end, progress=False)
    print(f"   ✓ Download concluído: {len(data)} registros")
    if not data.empty:
        print(data.head())
except Exception as e:
    print(f"   ✗ Erro: {e}")

# Teste 6: Múltiplos tickers
print("\n6. Testando múltiplos tickers...")
try:
    end = datetime.now()
    start = end - timedelta(days=30)
    data = yf.download(['PETR4.SA', 'VALE3.SA'], start=start, end=end, progress=False)
    print(f"   ✓ Download concluído")
    print(f"   Shape: {data.shape}")
    print(f"   Colunas: {data.columns.tolist()}")
    print(data.head())
except Exception as e:
    print(f"   ✗ Erro: {e}")

print("\n" + "="*60)
print("TESTE CONCLUÍDO")
print("="*60 + "\n")
