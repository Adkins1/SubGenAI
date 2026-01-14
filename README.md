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
z tlumaczeniem offline w tym samym UI. Wejscie pochodzi z /work/in (SMB share) albo z uploadu do kontenera.
Wyniki SRT trafiaja do tego samego folderu co plik zrodlowy.

## Najwazniejsze biblioteki
- faster-whisper (ASR)
- transformers + torch (tlumaczenie NLLB)
- FastAPI (API)
- Uvicorn (serwer ASGI)
- python-multipart (upload plikow)
- ffmpeg/ffprobe (czas trwania i obsluga mediow)

## Funkcje
- Dwa niezalezne procesy: Start transkrypcji i Start tlumaczenia.
- Wybierasz jezyk audio/wideo (Auto albo z listy) i model transkrypcji.
- Przycisk "Transkrybuj wszystko" iteruje po filmach w folderze i pomija te z gotowym SRT.
- Przy batchu widzisz licznik "Ukonczono X/Y" obok paska transkrypcji.
- Lista tlumaczen obejmuje wiele jezykow (Polski na gorze, reszta alfabetycznie).
- Nawigacja po folderach SMB, bez ladowania wszystkich plikow naraz.
- Dwa paski postepu, tabela statusow i przycisk Stop.
- Theme switcher (5 wariantow kolorystycznych).

## Struktura katalogow i zapis wynikow
- /work/in -> SMB share z mediami (czytanie + zapis SRT obok plikow).
- /work/uploads -> pliki wyslane przez UI.
- /work/out -> wolumen pomocniczy (obecnie nieuzywany).
- /hf -> cache modeli.
- Nazwy SRT: nazwa.<lang>.srt (np. .jp.srt, .pl.srt).

## Konfiguracja (.env)
Przyklad:
```env
SMB_SHARE=//NAS/Media
SMB_USER=twoj_user
SMB_PASS="TwojeHaslo!@#$"
SMB_OPTS=rw,vers=3.0,dir_mode=0775,file_mode=0664

TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# TORCH_VERSION=2.5.1
# CUDA_BASE=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
```

Uwaga: podmien SMB_* na swoje dane. Jesli haslo ma znaki specjalne, zostaw je w cudzyslowie.

## Domyslna karta graficzna
Docker udostepnia wszystkie GPU (gpus: all), a PyTorch uzywa domyslnie pierwszej karty widzianej przez CUDA (cuda:0).

## Usecase 1: standardowe uruchomienie i korzystanie <span style="color:#6fa8ff;">&#127799;</span>
1. Ustaw SMB_* w .env (share + login + haslo).
2. docker compose up -d --build
3. Otworz http://localhost:8000
4. Wybierz folder i plik w /work/in, ustaw model i jezyk audio, kliknij Start transkrypcji.
5. Wybierz plik SRT z listy i uruchom Start tlumaczenia.
6. Alternatywnie: uzyj "Transkrybuj wszystko", aby przetworzyc caly folder.

## Usecase 2: nieodpowiednia karta graficzna (CUDA) <span style="color:#ffb86b;">&#128161;</span>
Jesli widzisz ostrzezenia o nieobslugiwanej architekturze GPU albo fallback na CPU:
- Zmien TORCH_INDEX_URL w .env
- (Opcjonalnie) ustaw TORCH_VERSION i/lub CUDA_BASE
- Przebuduj obraz

Tabela (RTX 40xx i 50xx):

| Model | Arch | Zalecany TORCH_INDEX_URL | Dodatkowe uwagi |
|------|------|---------------------------|-----------------|
| RTX 4060 | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | Zostaw CUDA_BASE domyslne, chyba ze masz bledy builda |
| RTX 4060 Ti | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 4070 | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 4070 Super | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 4070 Ti | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 4070 Ti Super | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 4080 | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 4080 Super | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 4090 | sm_89 | https://download.pytorch.org/whl/cu121 (lub cu124/cu128) | jw. |
| RTX 5060 | sm_120 | https://download.pytorch.org/whl/cu128 (lub cu130) | Jesli nadal jest fallback, ustaw nowsze CUDA_BASE (np. 12.8.x/13.x) |
| RTX 5060 Ti | sm_120 | https://download.pytorch.org/whl/cu128 (lub cu130) | jw. |
| RTX 5070 | sm_120 | https://download.pytorch.org/whl/cu128 (lub cu130) | jw. |
| RTX 5070 Ti | sm_120 | https://download.pytorch.org/whl/cu128 (lub cu130) | jw. |
| RTX 5080 | sm_120 | https://download.pytorch.org/whl/cu128 (lub cu130) | jw. |
| RTX 5090 | sm_120 | https://download.pytorch.org/whl/cu128 (lub cu130) | jw. |

Po zmianach:
1. docker compose down
2. docker compose build --no-cache
3. docker compose up -d

## API (skrot)
- GET /api/files -> lista mediow z /work/in (z parametrem dir)
- GET /api/srt-files -> lista SRT z /work/in (z parametrem dir)
- GET /api/languages -> jezyki transkrypcji i tlumaczen
- POST /api/jobs -> start transkrypcji
- POST /api/translations -> start tlumaczenia
- POST /api/jobs/{id}/stop -> zatrzymanie joba
- GET /api/jobs/{id}/events -> SSE ze statusem
- GET /api/jobs/{id}/srt -> pobierz SRT oryginalny
- GET /api/jobs/{id}/srt-translated -> pobierz SRT tlumaczenie

<p style="margin-top:10px;">
  <span style="color:#ff6b6b;">&#10024;</span>
  <span style="color:#5adba0;">&#128155;</span>
  <span style="color:#6fa8ff;">&#128049;</span>
  <span style="color:#ffb86b;">&#127800;</span>
</p>
