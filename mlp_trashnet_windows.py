"""Unidad II - Clasificacion de residuos (TrashNet) con MLP. Preprocesamiento, baseline, ajuste de hiperparametros, evaluacion."""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import glob, json, random, numpy as np, cv2, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
SEED=42; random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"dataset-resized"); OUT=os.path.join(HERE,"resultados"); os.makedirs(OUT,exist_ok=True)
IMG=224; MLP_IMG=32
if not os.path.isdir(DATA):   # descarga automatica de TrashNet
    import urllib.request, zipfile
    z=os.path.join(HERE,"ds.zip"); print("Descargando TrashNet...")
    urllib.request.urlretrieve("https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",z)
    zipfile.ZipFile(z).extractall(HERE); print("Listo.")

# ---------- 1. PREPROCESAMIENTO (Fases 1-4 del pipeline del reporte) ----------
paths,labels=[],[]
for d in sorted(x for x in os.listdir(DATA) if os.path.isdir(os.path.join(DATA,x))):
    for f in glob.glob(os.path.join(DATA,d,"*.jpg")): paths.append(f); labels.append(d)
def preprocess(p):
    bgr=cv2.imread(p)                                   # Fase 1: ingesta (OpenCV lee BGR)
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)             # Fase 2: BGR->RGB
    rgb=cv2.resize(rgb,(IMG,IMG),interpolation=cv2.INTER_AREA)  # Fase 3: 224x224
    x=rgb.astype("float32")/255.0                       # Fase 4: normalizacion [0,1]
    x=cv2.resize(x,(MLP_IMG,MLP_IMG),interpolation=cv2.INTER_AREA) # reduccion para MLP (ver justificacion)
    return x.reshape(-1)                                # aplanado: 64*64*3 = 12288
X=np.stack([preprocess(p) for p in paths]); le=LabelEncoder(); y=le.fit_transform(labels)
classes=list(le.classes_); print("X",X.shape,"clases",classes)
# Particion estratificada 70/15/15
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,stratify=y,random_state=SEED)
Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=0.50,stratify=ytmp,random_state=SEED)
mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-6   # estandarizacion con estadisticas SOLO de train
Xtr,Xva,Xte=[(a-mu)/sd for a in (Xtr,Xva,Xte)]
np.savez(os.path.join(OUT,"preproc_stats.npz"),mu=mu,sd=sd,classes=np.array([str(c) for c in le.classes_]),img=MLP_IMG)  # para la Raspberry Pi
np.savez(os.path.join(OUT,"preprocess.npz"),mu=mu,sd=sd,classes=np.array(le.classes_),img=MLP_IMG)  # para usar el modelo con la camara
print("train/val/test",len(Xtr),len(Xva),len(Xte))
cw=dict(enumerate(compute_class_weight("balanced",classes=np.unique(ytr),y=ytr)))  # desbalance de 'trash'
print("class_weight",{classes[k]:round(v,2) for k,v in cw.items()})

# ---------- 2. ARQUITECTURA MLP ----------
def build(lr=1e-3,units=(256,128),drop=0.3,l2=1e-4):
    reg=tf.keras.regularizers.l2(l2)
    L=[tf.keras.layers.Input((X.shape[1],))]
    for u in units:
        L+= [tf.keras.layers.Dense(u,activation="relu",kernel_regularizer=reg),tf.keras.layers.Dropout(drop)]
    L+=[tf.keras.layers.Dense(len(classes),activation="softmax")]
    m=tf.keras.Sequential(L)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return m
def train(lr,batch=32,epochs=40,tag=""):
    tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
    m=build(lr=lr)
    es=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True)
    h=m.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=epochs,batch_size=batch,class_weight=cw,callbacks=[es],verbose=0)
    va=max(h.history["val_accuracy"]); print(f"{tag} lr={lr} batch={batch}: epocas={len(h.history['loss'])} val_acc={va:.4f}")
    return m,h
_lines=[]
build().summary(print_fn=lambda s:_lines.append(s))
open(f"{OUT}/model_summary.txt","w",encoding="utf-8").write("\n".join(_lines))

# ---------- 3. BASELINE + 4. AJUSTE DE LEARNING RATE ----------
results={}; models={}
for lr in [1e-2,1e-3,1e-4]:
    m,h=train(lr,tag="LR"); models[lr]=(m,h)
    results[lr]=dict(epochs=len(h.history["loss"]),best_val_acc=float(max(h.history["val_accuracy"])),
                     best_val_loss=float(min(h.history["val_loss"])),final_train_acc=float(h.history["accuracy"][-1]))
# batch size (segundo hiperparametro, sobre el mejor LR)
best_lr=min(results,key=lambda k:results[k]["best_val_loss"])
batch_res={}
for b in [16,64]:
    m,h=train(best_lr,batch=b,tag="BATCH"); models[("b",b)]=(m,h)
    batch_res[b]=dict(epochs=len(h.history["loss"]),best_val_acc=float(max(h.history["val_accuracy"])),best_val_loss=float(min(h.history["val_loss"])))

# ---------- 5. EVALUACION en test (baseline = lr 1e-3, batch 32) ----------
def evaluate(m,name):
    p=m.predict(Xte,verbose=0).argmax(1); acc=accuracy_score(yte,p)
    return p,acc
cands={f"lr={k}":v[0] for k,v in models.items() if not isinstance(k,tuple)}
cands.update({f"lr={best_lr},batch={k[1]}":v[0] for k,v in models.items() if isinstance(k,tuple)})
test_acc={n:float(evaluate(m,n)[1]) for n,m in cands.items()}; print("test acc",test_acc)
base=models[1e-3][0]; pb,accb=evaluate(base,"baseline")
rep=classification_report(yte,pb,target_names=classes,output_dict=True); print(classification_report(yte,pb,target_names=classes))
cm=confusion_matrix(yte,pb)
# mejor modelo segun validacion
allv={("lr",k):v["best_val_acc"] for k,v in results.items()}; allv.update({("b",k):v["best_val_acc"] for k,v in batch_res.items()})
bestk=max(allv,key=allv.get); bm=models[bestk[1] if bestk[0]=="lr" else ("b",bestk[1])][0]
pbest,accbest=evaluate(bm,"best"); cmb=confusion_matrix(yte,pbest)
repb=classification_report(yte,pbest,target_names=classes,output_dict=True)

# ---------- FIGURAS ----------
def curves(h,fn,title):
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    ax[0].plot(h.history["loss"],label="train");ax[0].plot(h.history["val_loss"],label="val");ax[0].set_title("Perdida");ax[0].set_xlabel("Epoca");ax[0].legend()
    ax[1].plot(h.history["accuracy"],label="train");ax[1].plot(h.history["val_accuracy"],label="val");ax[1].set_title("Exactitud");ax[1].set_xlabel("Epoca");ax[1].legend()
    fig.suptitle(title);fig.tight_layout();fig.savefig(fn,dpi=130);plt.close(fig)
curves(models[1e-3][1],f"{OUT}/fig_curvas_baseline.png","Baseline (lr=1e-3, batch=32)")
def plotcm(cm,fn,title):
    fig,ax=plt.subplots(figsize=(5.5,4.8)); ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xticklabels(classes,rotation=45,ha="right");ax.set_yticklabels(classes)
    for i in range(6):
        for j in range(6): ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xlabel("Predicha");ax.set_ylabel("Real");ax.set_title(title);fig.tight_layout();fig.savefig(fn,dpi=130);plt.close(fig)
plotcm(cm,f"{OUT}/fig_matriz_confusion_baseline.png","Matriz de confusion - baseline (test)")
fig,ax=plt.subplots(figsize=(6,3.6))
for lr in [1e-2,1e-3,1e-4]: ax.plot(models[lr][1].history["val_loss"],label=f"lr={lr}")
ax.set_xlabel("Epoca");ax.set_ylabel("Perdida validacion");ax.set_title("Efecto de la tasa de aprendizaje");ax.legend();fig.tight_layout();fig.savefig(f"{OUT}/fig_lr_comparacion.png",dpi=130);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3.2)); cnt=np.bincount(y)
ax.bar(classes,cnt);ax.set_title("Distribucion de clases (TrashNet)");fig.tight_layout();fig.savefig(f"{OUT}/fig_distribucion.png",dpi=130);plt.close(fig)
base.save(f"{OUT}/mlp_baseline.keras"); bm.save(f"{OUT}/mlp_mejor.keras")
json.dump(dict(classes=classes,split=[len(Xtr),len(Xva),len(Xte)],class_weight={classes[k]:v for k,v in cw.items()},
  lr=results,batch=batch_res,test_acc=test_acc,baseline_report=rep,baseline_cm=cm.tolist(),best=str(bestk),best_acc=accbest if False else float(accbest),
  best_report=repb,best_cm=cmb.tolist(),params=int(base.count_params())),open(f"{OUT}/resultados.json","w",encoding="utf-8"),indent=1,default=str)

# ---------- FIGURAS ADICIONALES ----------
# 1) Muestras del dataset: original, 224x224 (RGB) y 32x32 usada por el MLP
fig,ax=plt.subplots(3,6,figsize=(12,6.4))
for j,c in enumerate(classes):
    p=next(q for q,l in zip(paths,labels) if l==c)
    rgb=cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
    r224=cv2.resize(rgb,(IMG,IMG),interpolation=cv2.INTER_AREA)
    r32=cv2.resize(r224,(MLP_IMG,MLP_IMG),interpolation=cv2.INTER_AREA)
    for i,(im,t) in enumerate([(rgb,"original"),(r224,"224x224"),(r32,f"{MLP_IMG}x{MLP_IMG} (MLP)")]):
        ax[i,j].imshow(im,interpolation="nearest"); ax[i,j].axis("off")
        ax[i,j].set_title(f"{c}\n{t}" if i==0 else t,fontsize=8)
fig.suptitle("Preprocesamiento: de la imagen original a la entrada del MLP"); fig.tight_layout()
fig.savefig(f"{OUT}/fig_muestras_preprocesamiento.png",dpi=130); plt.close(fig)
# 2) Comparacion de tasa de aprendizaje y lote (exactitud de validacion)
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].bar([str(k) for k in results],[v["best_val_acc"] for v in results.values()]); ax[0].set_title("Val. accuracy vs tasa de aprendizaje"); ax[0].set_ylim(0,1)
ax[1].bar([f"lote {k}" for k in batch_res],[v["best_val_acc"] for v in batch_res.values()]); ax[1].set_title(f"Val. accuracy vs lote (lr={best_lr})"); ax[1].set_ylim(0,1)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_hiperparametros.png",dpi=130); plt.close(fig)
# 3) Matriz de confusion normalizada (recall por clase)
cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(5.8,5)); im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xticklabels(classes,rotation=45,ha="right");ax.set_yticklabels(classes)
for i in range(6):
    for j in range(6): ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",color="white" if cmn[i,j]>0.5 else "black")
ax.set_xlabel("Predicha");ax.set_ylabel("Real");ax.set_title("Matriz de confusion normalizada (baseline, test)")
fig.colorbar(im);fig.tight_layout();fig.savefig(f"{OUT}/fig_matriz_confusion_normalizada.png",dpi=130);plt.close(fig)
# 4) Curvas del mejor modelo por validacion
print("\nImagenes y archivos generados en:",OUT)
for f in sorted(os.listdir(OUT)): print("  -",f)
