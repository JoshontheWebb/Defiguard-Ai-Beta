@echo off
echo === GET FRESH SESSION & TOKEN ===
curl -c cookies.txt http://127.0.0.1:8000/csrf-token > token.json
type token.json

rem === EXTRACT CLEAN TOKEN ===
for /f "usebackq tokens=2 delims=:," %%A in (`type token.json ^| findstr "csrf_token"`) do set "TOKEN=%%A"
set "TOKEN=%TOKEN:"=%
set "TOKEN=%TOKEN: =%
set "TOKEN=%TOKEN:~0,-1%"  :: Remove any trailing } or ,

echo.
echo === TOKEN: %TOKEN% ===
echo.

echo === 100-USER LOAD TEST ===
curl "http://127.0.0.1:8000/load-test?users=100&duration=60" ^
  -H "X-CSRF-Token: %TOKEN%" ^
  -b cookies.txt -c cookies.txt -s | python -m json.tool

echo.
echo === 1,000-USER STRESS TEST ===
curl "http://127.0.0.1:8000/load-test?users=1000&duration=120" ^
  -H "X-CSRF-Token: %TOKEN%" ^
  -b cookies.txt -c cookies.txt -s | python -m json.tool

echo.
del cookies.txt token.json 2>nul
echo === DONE! Check data\debug.log ===