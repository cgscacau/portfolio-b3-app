# Correções Necessárias

## 1. Remover st.switch_page() do run_app.py

Já corrigido acima - remover os botões com st.switch_page() da home.

## 2. Ajustar paths em todas as páginas

Em TODAS as páginas (01 a 05), substituir:
```python
st.switch_page("app/pages/XX_Nome.py")
