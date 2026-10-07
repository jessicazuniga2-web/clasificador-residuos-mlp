"""Clasificacion de residuos con la camara de la computadora (usa el MLP entrenado).
Uso:
  python clasificar_camara.py                 -> camara en vivo
  python clasificar_camara.py --imagen foto.jpg  -> clasifica una imagen (sin camara)
Teclas (camara): ESPACIO = congelar/guardar captura | Q o ESC = salir
"""
import os, sys, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, cv2, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "resultados")
UMBRAL = 0.60  # RF-05: alerta si la confianza maxima es menor al 60 %

# Nombre mostrado y contenedor SUGERIDO (colores orientativos; ajusta a tu institucion)
INFO = {
    "cardboard": ("Carton",        "Azul",    (255, 120, 0)),
    "glass":     ("Vidrio",        "Verde",   (0, 170, 0)),
    "metal":     ("Metal",         "Amarillo",(0, 200, 255)),
    "paper":     ("Papel",         "Azul",    (255, 120, 0)),
    "plastic":   ("Plastico",      "Amarillo",(0, 200, 255)),
    "trash":     ("Basura general","Gris",    (150, 150, 150)),
}

def cargar():
    ruta_modelo = os.path.join(RES, "mlp_baseline.keras")
    ruta_pre = os.path.join(RES, "preprocess.npz")
    for r in (ruta_modelo, ruta_pre):
        if not os.path.exists(r):
            sys.exit(f"No se encontro {r}.\nEjecuta primero mlp_trashnet_windows.py o copia la carpeta 'resultados'.")
    modelo = tf.keras.models.load_model(ruta_modelo)
    p = np.load(ruta_pre, allow_pickle=True)
    return modelo, p["mu"], p["sd"], [str(c) for c in p["classes"]], int(p["img"])

def recorte_central(bgr):
    h, w = bgr.shape[:2]; l = min(h, w); y0, x0 = (h - l) // 2, (w - l) // 2
    return bgr[y0:y0 + l, x0:x0 + l]

def preprocesar(bgr, mu, sd, m):
    """Mismas fases que el entrenamiento: BGR->RGB, 224x224, [0,1], reduccion, estandarizacion."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    x = rgb.astype("float32") / 255.0
    x = cv2.resize(x, (m, m), interpolation=cv2.INTER_AREA).reshape(1, -1)
    return (x - mu) / sd

def predecir(modelo, bgr, mu, sd, clases, m):
    probs = modelo.predict(preprocesar(bgr, mu, sd, m), verbose=0)[0]
    i = int(probs.argmax()); return clases[i], float(probs[i]), probs

def texto_resultado(clase, conf):
    nombre, cont, _ = INFO.get(clase, (clase, "?", (255, 255, 255)))
    if conf < UMBRAL:
        return f"Incierto ({conf*100:.0f}%) - toma otra foto", (0, 0, 255), None
    return f"{nombre} {conf*100:.0f}% -> contenedor {cont}", INFO[clase][2], nombre

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--imagen"); ap.add_argument("--camara", type=int, default=0)
    a = ap.parse_args(); modelo, mu, sd, clases, m = cargar()
    if a.imagen:
        bgr = cv2.imread(a.imagen)
        if bgr is None: sys.exit("No se pudo leer la imagen.")
        clase, conf, probs = predecir(modelo, recorte_central(bgr), mu, sd, clases, m)
        print(texto_resultado(clase, conf)[0])
        for c, p in sorted(zip(clases, probs), key=lambda t: -t[1]): print(f"  {INFO[c][0]:15s} {p*100:5.1f}%")
        return
    cap = cv2.VideoCapture(a.camara, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.camara)
    if not cap.isOpened(): sys.exit("No se pudo abrir la camara. Prueba con --camara 1.")
    congelado, cuenta = None, 0
    print("ESPACIO = congelar/guardar captura | Q o ESC = salir")
    while True:
        ok, frame = cap.read()
        if not ok: break
        vista = congelado if congelado is not None else frame
        roi = recorte_central(vista)
        clase, conf, _ = predecir(modelo, roi, mu, sd, clases, m)
        txt, color, _ = texto_resultado(clase, conf)
        h, w = vista.shape[:2]; l = min(h, w); x0, y0 = (w - l) // 2, (h - l) // 2
        out = vista.copy()
        cv2.rectangle(out, (x0, y0), (x0 + l, y0 + l), color, 3)       # zona que analiza el modelo
        cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(out, txt, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if congelado is not None: cv2.putText(out, "CAPTURA (ESPACIO para continuar)", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Clasificador de residuos", out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27): break
        if k == 32:
            if congelado is None:
                congelado = frame.copy(); cuenta += 1
                os.makedirs(os.path.join(HERE, "capturas"), exist_ok=True)
                cv2.imwrite(os.path.join(HERE, "capturas", f"captura_{cuenta:03d}_{clase}.jpg"), out)
            else: congelado = None
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
