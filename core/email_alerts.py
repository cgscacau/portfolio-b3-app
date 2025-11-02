"""
Módulo de envio de alertas por email usando SendGrid
"""

import streamlit as st
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)


def enviar_alerta_oportunidades(oportunidades: List[Dict]) -> bool:
    """
    Envia email com oportunidades detectadas
    
    Args:
        oportunidades: Lista de dicionários com dados das oportunidades
        
    Returns:
        True se enviado com sucesso, False caso contrário
    """
    
    try:
        # Buscar credenciais do Streamlit Secrets
        api_key = st.secrets["email"]["sendgrid_api_key"]
        email_from = st.secrets["email"]["email_from"]
        email_to = st.secrets["email"]["email_to"]
        
    except Exception as e:
        logger.error(f"Erro ao carregar credenciais: {str(e)}")
        st.error("⚠️ Configure os Secrets no Streamlit Cloud (Settings → Secrets)")
        return False
    
    if not oportunidades:
        logger.info("Nenhuma oportunidade para enviar")
        return False
    
    # Construir HTML do email
    html_content = construir_html_email(oportunidades)
    
    # Criar mensagem
    message = Mail(
        from_email=Email(email_from),
        to_emails=To(email_to),
        subject=f"🎯 Cacau's Channel - {len(oportunidades)} Oportunidade(s) Detectada(s)",
        html_content=Content("text/html", html_content)
    )
    
    try:
        # Enviar email
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        
        logger.info(f"Email enviado com sucesso! Status: {response.status_code}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao enviar email: {str(e)}")
        return False


def construir_html_email(oportunidades: List[Dict]) -> str:
    """
    Constrói HTML do email com tabela de oportunidades
    
    Args:
        oportunidades: Lista de oportunidades
        
    Returns:
        String HTML
    """
    
    # Cabeçalho
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .timestamp {
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th {
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .compra {
                color: #27ae60;
                font-weight: bold;
            }
            .venda {
                color: #e74c3c;
                font-weight: bold;
            }
            .footer {
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
                font-size: 12px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Cacau's Channel - Oportunidades Detectadas</h1>
            <p class="timestamp">📅 """ + datetime.now().strftime('%d/%m/%Y às %H:%M:%S') + """</p>
            
            <p>Foram detectadas <strong>""" + str(len(oportunidades)) + """</strong> oportunidade(s) com convergência entre timeframes diário e semanal:</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Ativo</th>
                        <th>Direção</th>
                        <th>Entrada</th>
                        <th>Stop Loss</th>
                        <th>Alvo</th>
                        <th>R/R</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Adicionar linhas da tabela
    for opp in oportunidades:
        direcao_class = "compra" if opp['direcao'] == 'COMPRA' else "venda"
        
        html += f"""
                    <tr>
                        <td><strong>{opp['ticker']}</strong></td>
                        <td class="{direcao_class}">{opp['direcao']}</td>
                        <td>R$ {opp['entrada']:.2f}</td>
                        <td>R$ {opp['stop']:.2f}</td>
                        <td>R$ {opp['alvo']:.2f}</td>
                        <td>{opp['rr']}</td>
                    </tr>
        """
    
    # Rodapé
    html += """
                </tbody>
            </table>
            
            <div class="footer">
                <p>Este é um alerta automático gerado pelo sistema Cacau's Channel.</p>
                <p>⚠️ Esta não é uma recomendação de investimento. Faça sua própria análise.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def testar_configuracao_email() -> bool:
    """
    Testa se as configurações de email estão corretas
    
    Returns:
        True se configurado corretamente
    """
    try:
        api_key = st.secrets["email"]["sendgrid_api_key"]
        email_from = st.secrets["email"]["email_from"]
        email_to = st.secrets["email"]["email_to"]
        
        if not api_key or not email_from or not email_to:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao verificar configuração: {str(e)}")
        return False


def enviar_email_teste() -> bool:
    """
    Envia email de teste
    
    Returns:
        True se enviado com sucesso
    """
    try:
        api_key = st.secrets["email"]["sendgrid_api_key"]
        email_from = st.secrets["email"]["email_from"]
        email_to = st.secrets["email"]["email_to"]
        
    except Exception as e:
        logger.error(f"Erro ao carregar credenciais: {str(e)}")
        return False
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .container { max-width: 600px; margin: 0 auto; background-color: #f9f9f9; padding: 30px; border-radius: 10px; }
            h1 { color: #2c3e50; }
            .success { color: #27ae60; font-size: 18px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✅ Teste de Configuração</h1>
            <p class="success">Parabéns! Seu sistema de alertas está configurado corretamente.</p>
            <p>Você receberá emails automáticos sempre que o sistema detectar oportunidades no Cacau's Channel.</p>
            <p><strong>Data/Hora:</strong> """ + datetime.now().strftime('%d/%m/%Y às %H:%M:%S') + """</p>
        </div>
    </body>
    </html>
    """
    
    message = Mail(
        from_email=Email(email_from),
        to_emails=To(email_to),
        subject="✅ Teste - Cacau's Channel",
        html_content=Content("text/html", html_content)
    )
    
    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        
        logger.info(f"Email de teste enviado! Status: {response.status_code}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao enviar email de teste: {str(e)}")
        return False
