# Página de flares en el portal: 8 parches para `clima-espacial`

Estos parches pasan el producto `usach.flare.nrt` a Centella y le dan su página. Se aplican en orden sobre `main`. Las decisiones están explicadas en `docs/DECISIONS.md` (D-099 a D-103), y `DESCRIPCION-PR.md` sirve tal cual como descripción del PR.

```bash
cd clima-espacial
git fetch origin
git checkout -b feat/flare-nrt-v2 origin/main
git am /ruta/a/3-portal-clima-espacial/*.patch
ruff check . && pytest
git push -u origin feat/flare-nrt-v2       # y abrir el PR con DESCRIPCION-PR.md
```

Si `main` avanzó y un parche choca, `git am` se detiene en ese parche: se resuelve el conflicto, se hace `git add` y luego `git am --continue`. El conflicto más probable es en `docs/DECISIONS.md`, que solo recibe líneas al final.

Al desplegar:
- `FLARE_NRT_URL` es la API de Centella. Por defecto apunta a `http://127.0.0.1:5050`, que es lo que deja instalado `deploy/README.md` de flare-nrt.
- `FLARE_ARCHIVE_URL` es opcional: una API de Centella sobre un período reproducido. Si no está definida, la página no muestra el botón "Archivo".
- No trae migraciones ni dependencias nuevas.
