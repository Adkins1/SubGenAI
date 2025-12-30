# <span style="color:#ff6b6b;">SUB</span><span style="color:#5adba0;">GEN</span><span style="color:#6fa8ff;">AI</span>

<p style="margin-top:6px; font-size:14px;">
  Lokalny generator napisow SRT (transkrypcja + tlumaczenie) w jednym kontenerze GPU.
</p>

<p>
  <span style="display:inline-block; padding:4px 10px; border:1px solid #2a2f3a; border-radius:999px; margin-right:6px;">
    &#9889; GPU-first
  </span>
  <span style="display:inline-block; padding:4px 10px; border:1px solid #2a2f3a; border-radius:999px; margin-right:6px;">
    &#127916; video/audio
  </span>
  <span style="display:inline-block; padding:4px 10px; border:1px solid #2a2f3a; border-radius:999px; margin-right:6px;">
    &#128187; lokalnie
  </span>
  <span style="display:inline-block; padding:4px 10px; border:1px solid #2a2f3a; border-radius:999px;">
    &#128196; SRT
  </span>
</p>

## Opis
SUBGENAI to aplikacja web do lokalnej transkrypcji wideo/audio i generowania napisow SRT w kontenerze GPU,
z opcjonalnym tlumaczeniem na polski (offline) w tym samym UI. Wejscie pochodzi z /work/in (mount z hosta)
lub z uploadu do kontenera. Wyniki trafiaja do /work/out.

## Najwazniejsze biblioteki
- faster-whisper (ASR)
- transformers + torch (tlumaczenie NLLB)
- FastAPI (API)
- Uvicorn (serwer ASGI)
- python-multipart (upload plikow)
- ffmpeg/ffprobe (czas trwania i obsluga mediow)

## Co nowego
- Rozdzielone procesy: osobno Start transkrypcji i Start tlumaczenia.
- Tlumaczenie wybiera konkretny plik SRT z /work/out.
- Dwa paski postepu + tabela statusow pod spodem.
- Przycisk Stop zatrzymuje aktywny proces (transkrypcja lub tlumaczenie).
- Linki do pobrania: SRT oryginalny oraz SRT tlumaczony.

## UI - jak dziala
- Lista plikow: odswiezanie listy z /work/in i wybor pliku.
- Upload: opcjonalny, zapisuje plik do /work/uploads w kontenerze.
- Start transkrypcji: wysyla job do API, UI pokazuje pasek postepu, status i timer.
- Start tlumaczenia: wybor jezyka (na razie tylko polski) + wybor SRT z /work/out.
- Dwa paski postepu (transkrypcja / tlumaczenie) i tabela statusow (czas, model, jezyk, progres).
- Stop: zatrzymuje aktywny job.
- Po zakonczeniu widoczne sa linki do pobrania SRT oryginalnego i tlumaczonego.

## Uruchomienie
1. Ustaw FILES_DIR w .env (sciezka na hoscie do plikow z wideo/audio).
2. docker compose up --build
3. Otworz http://localhost:8000

Przyklad .env:
```env
FILES_DIR=C:/Media
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# TORCH_VERSION=2.5.1
# CUDA_BASE=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
```

## Struktura katalogow i nazwy plikow
- /work/in -> katalog z mediami (read-only, z hosta).
- /work/out -> zapis SRT na hosta (./out).
- /work/uploads -> pliki wyslane przez UI.
- /hf -> cache modeli.
- Nazwy SRT: nazwa.<lang>.srt (np. .jp.srt, .pl.srt).

## Wskazowki GPU
- Tlumaczenie korzysta z GPU tylko gdy PyTorch obsluguje Twoja architekture CUDA.
- Jesli widzisz fallback na CPU, ustaw TORCH_INDEX_URL na nowsze wheel'e (np. cu128 lub cu130)
  i przebuduj obraz.

## API (skrot)
- GET /api/files -> lista mediow z /work/in
- GET /api/srt-files -> lista SRT z /work/out
- POST /api/jobs -> start transkrypcji
- POST /api/translations -> start tlumaczenia
- POST /api/jobs/{id}/stop -> zatrzymanie joba
- GET /api/jobs/{id}/events -> SSE ze statusem
- GET /api/jobs/{id}/srt -> pobierz SRT oryginalny
- GET /api/jobs/{id}/srt-translated -> pobierz SRT tlumaczony
