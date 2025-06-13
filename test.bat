@echo off
echo [*] Sending request to Orpheus...

curl -X POST http://localhost:5005/speak ^
  -H "Content-Type: application/json" ^
  -d "{ \"text\": \"This is a test from Orpheus TTS.\", \"voice\": \"tara\" }" ^
  -o output.wav

if exist output.wav (
  echo [*] Playing audio...
  powershell -c (New-Object Media.SoundPlayer "output.wav").PlaySync()
) else (
  echo [!] Audio file not found!
)
pause
