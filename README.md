# fw-web

## Opis
Aplikacja web do lokalnej transkrypcji wideo/audio i generowania napisow SRT w kontenerze GPU. Wejscie pochodzi z /work/in (mount z hosta) lub z uploadu do kontenera. Wyniki trafiaja do /work/out.

## Najwazniejsze biblioteki
- faster-whisper (ASR)
- FastAPI (API)
- Uvicorn (serwer ASGI)
- python-multipart (upload plikow)
- ffmpeg/ffprobe (czas trwania i obsluga mediow)

## UI - jak dziala
- Lista plikow: odswiezanie listy z /work/in i wybor pliku.
- Upload: opcjonalny, zapisuje plik do /work/uploads w kontenerze.
- Start transkrypcji: wysyla job do API, UI pokazuje pasek postepu, status i timer.
- Wykryty jezyk i nazwa wyjsciowego SRT pojawiaja sie w trakcie.
- Po zakonczeniu widoczny jest link do pobrania SRT.

## Uruchomienie
1. Ustaw FILES_DIR w .env (sciezka na hoscie do plikow z wideo/audio).
2. docker compose up --build
3. Otworz http://localhost:8000

Dodatkowo:
- /work/in -> katalog z mediami (read-only).
- /work/out -> zapis SRT na hosta (./out).
