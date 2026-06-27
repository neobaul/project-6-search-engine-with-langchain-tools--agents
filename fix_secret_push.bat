@echo off
cd /d "F:\GEN_AI\final submission\project 6 Search engine with langchain tools & agents"

echo Wiping git history...
rmdir /s /q .git

echo Reinitializing git...
git init
git branch -M main

echo Staging files...
git add .

echo Committing...
git commit -m "Initial commit"

echo Adding remote...
git remote add origin https://github.com/neobaul/project-6-search-engine-with-langchain-tools--agents.git

echo Pushing...
git push -u origin main --force

echo Done!
pause
