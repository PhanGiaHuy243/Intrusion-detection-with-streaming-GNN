@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   DO AN STREAMING GNN - PHAT HIEN XAM NHAP
echo ============================================
docker --version >nul 2>&1
if errorlevel 1 (
  echo [LOI] Chua cai Docker Desktop.
  echo Tai tai: https://www.docker.com/products/docker-desktop/
  pause
  exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
  echo [LOI] Docker Desktop chua mo. Hay mo Docker Desktop, doi no chay xong roi chay lai file nay.
  pause
  exit /b 1
)
echo [1/2] Dang dung va khoi dong (lan dau mat vai phut de tai thu vien)...
docker compose up -d --build
if errorlevel 1 (
  echo [LOI] Khong khoi dong duoc. Xem thong bao loi o tren.
  pause
  exit /b 1
)
echo [2/2] Dang doi ung dung san sang...
for /l %%i in (1,1,60) do (
  curl -fs http://localhost:8501/_stcore/health >nul 2>&1 && goto ready
  timeout /t 2 /nobreak >nul
)
:ready
echo.
echo XONG! Dang mo trinh duyet: http://localhost:8501
start "" http://localhost:8501
echo Muon tat ung dung: chay file dung_docker.bat
pause
