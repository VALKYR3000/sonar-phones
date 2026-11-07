# Sonar Phones

Sonar acústico con varios celulares. Cada celular abre una página web, emite y graba chirps ultrasónicos (10–14 kHz) por turnos, y un servidor Python:

1. mide las **distancias entre celulares** (método BeepBeep, sin sincronizar relojes),
2. reconstruye sus **posiciones 2D/3D** (MDS + SMACOF ponderado),
3. genera un **mapa de ecos** (retroproyección) de lo que hay en la sala,
4. detecta **enlaces bloqueados** (tomografía por atenuación) para ver personas/objetos entre celulares.

## Estructura

| Archivo | Función |
|---|---|
| `server.py` | Servidor aiohttp (HTTPS + WebSocket) y todo el procesamiento de señal |
| `static/index.html` | Página "nodo" que se abre en cada celular (emite/graba audio) |
| `static/radar.html` | Visualizador 3D ("Sonar Vision 3D") del mapa y los enlaces |
| `test_sim.py` | Simulación de una habitación 6×4.5×2.6 m con 5 celulares; mide el error de distancias y posición |
| `requirements.txt` | Dependencias |

## Uso

```bash
pip install -r requirements.txt
python3 server.py [--port 8443] [--interval 10] [--temp 20] [--debug]
```

- `--interval N`: mide automáticamente cada N s (0 = manual, por defecto).
- `--temp`: temperatura en °C para la velocidad del sonido (`c = 331.3 + 0.606·T`).
- `--debug`: guarda `last_rec.npz` con la última grabación.

Al arrancar, el servidor genera un certificado autofirmado (`openssl`) porque los navegadores solo dan acceso al micrófono por HTTPS. Luego:

1. En **cada celular** abre `https://<IP-del-PC>:8443`, acepta la advertencia del certificado, escribe un nombre y pulsa **Iniciar nodo** (permite el micrófono). Mínimo 2 celulares, ideal 4+.
2. Visualización: `https://<IP>:8443/radar`.
3. Medir manualmente: `curl -k https://localhost:8443/api/measure`

## API

| Ruta | Descripción |
|---|---|
| `/ws` | WebSocket de los nodos (`hello`, `meta`, audio binario) |
| `/api/nodes` | Nombres de los nodos conectados |
| `/api/measure` | Lanza una ronda y devuelve distancias, posiciones, mapa y enlaces |
| `/api/last` | Último resultado |
| `/api/calibrate` | Reinicia la línea base de atenuación (dejar la sala libre ~8 rondas) |

## Cómo funciona

- **Señal**: chirp lineal 10→14 kHz de 50 ms. Cada nodo emite un tren de 5 chirps con separaciones de regla de Golomb (`OFFS = 0, 0.1, 0.4, 0.9, 1.1 s`), de modo que la suma de la envolvente correlacionada en esos desfases da un pico muy robusto frente al ruido.
- **Turnos**: cada celular emite en su *slot* (cada `GAP = 2 s`) mientras todos graban.
- **Distancias (BeepBeep)**: con los instantes de llegada del chirp propio y ajeno en ambas grabaciones, `d = |(t_ab − t_aa) + (t_ba − t_bb)|·c/2`; elimina el desfase de reloj y la latencia de audio. Se usa la mediana de las últimas 9 rondas. Se descartan picos con SNR < 35 y distancias > 20 m.
- **Posiciones**: matriz de distancias (con huecos) → camino mínimo → MDS → SMACOF ponderado. Si el resultado es casi coplanar, se fuerza `z = 0` y el mapa es plano. El marco de referencia se mantiene entre rondas (Procrustes).
- **Mapa de ecos**: retroproyección sobre una rejilla 3D usando el tiempo de vuelo emisor→punto→receptor; se acumula `log1p` y se suaviza en el tiempo.
- **Bloqueos**: para cada enlace se compara la amplitud del sonido directo contra su historial (percentil 80) y la atenuación en dB ilumina una franja elíptica entre los dos celulares.

## Pruebas

```bash
python3 test_sim.py
```

Simula 5 celulares y 6 superficies con eco, e imprime el error máximo de distancia, el error de posición 3D y qué fracción de los puntos más fuertes cae cerca de objetos reales. Guarda una figura en `/tmp/sim3d.png`.

## Seguridad y privacidad

- `key.pem`/`cert.pem` **no se versionan**; se regeneran solos. El servidor escucha en `0.0.0.0`: úsalo solo en una red de confianza.
- Los nodos graban audio del micrófono; las grabaciones (`*.npz`) tampoco se versionan.

## Limitaciones conocidas

- La precisión depende de la latencia/estabilidad de audio de cada navegador y celular.
- Los parámetros (`MIN_SNR`, `MARGIN`, `POW`…) están afinados empíricamente en `server.py`.
