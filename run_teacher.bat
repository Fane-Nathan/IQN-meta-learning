@echo off
echo 🎓 Teacher: Waking up...
echo 📊 Step 1: Gathering latest student data (Merging Rollouts)...
"C:\Users\felix\Documents\linesight\venv\Scripts\python.exe" scripts/merge_rollouts.py

echo 🧠 Step 2: Analyzing performance and updating curriculum...
"C:\Users\felix\Documents\linesight\venv\Scripts\python.exe" scripts/curriculum_manager.py

echo ✅ Lesson Plan Updated!
echo Check 'curriculum_config.txt' for the new spawn zone.
pause
