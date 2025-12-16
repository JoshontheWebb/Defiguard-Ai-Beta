@echo off 
set PYTHONPATH=. 
for /f "tokens=*" %%i in ('venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv('C:\\\\Users\\\\client\\\\DeFiGuard\\\\.env'); import os; print('\n'.join([f'set {k}={v}' for k, v in os.environ.items() if k in ['GROK_API_KEY', 'INFURA_PROJECT_ID', 'STRIPE_API_KEY', 'STRIPE_WEBHOOK_SECRET', 'STRIPE_PRICE_PRO', 'STRIPE_PRICE_BEGINNER', 'STRIPE_PRICE_DIAMOND']]))"') do set %%i 
