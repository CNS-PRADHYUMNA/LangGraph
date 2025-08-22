@echo off

echo Checking git status...
git status

if "%~1"=="" (
  echo Please provide a commit message.
  pause
  exit /b
)

echo Adding all files...
git add .

echo Committing changes...
git commit -m "%~1"

echo Pushing to GitHub...
git push

echo Checking git status...
git status

pause
