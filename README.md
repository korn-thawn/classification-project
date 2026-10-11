# OK/NG Camera Vision PoC

ระบบตรวจชิ้นงาน OK/NG ด้วย webcam + CNN ขนาดเล็กที่ **train from scratch** (ไม่ใช้ pretrained weights ไม่ใช้โมเดลสำเร็จรูป) ใช้ทดลองบนโน๊ตบุ๊คก่อนขยายไปใช้งานจริงในโรงงาน

- ทุก dependency เป็น license MIT / BSD / Apache-2.0 (ดูตารางด้านล่างและหมายเหตุ)
- ไม่มีโค้ดส่วนไหนดาวน์โหลด weights หรือ dataset จาก internet
- preprocessing และ augmentation เขียนเองด้วย NumPy + OpenCV

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `config.yaml` | ค่าทั้งหมดที่ปรับได้ (ROI, ขนาดภาพ, augmentation, epoch, lr, threshold ฯลฯ) |
| `capture.py` | ถ่ายภาพจาก webcam: `o` = OK, `n` = NG, `q` = ออก |
| `preview_augment.py` | บันทึกตัวอย่างภาพหลัง augment ไว้ดูด้วยตา |
| `train.py` | train โมเดล (ใช้ GPU ถ้ามี), class weight, early stopping |
| `evaluate.py` | confusion matrix, NG recall, false reject rate, เลือก threshold |
| `server.py` | FastAPI: `POST /v1/inspect`, `GET /v1/health`, `GET /` |
| `vision/preprocess.py` | crop ROI → resize → normalize (ใช้ร่วมกันทั้งตอน train และ inference) |
| `vision/augment.py` | flip, rotate/translate/scale, brightness, contrast, gamma, noise, blur |
| `vision/model.py` | `SmallCNN`: 5 conv block + BatchNorm + global average pooling |
| `vision/dataset.py` | แบ่ง train/val/test **ตาม session** (กัน data leakage) |
| `vision/inference.py` | โหลด checkpoint และคำนวณ score (ใช้ร่วมกันระหว่าง evaluate และ server) |
| `tools/make_synthetic_data.py` | สร้างภาพจำลองไว้ทดสอบ pipeline โดยไม่ต้องใช้กล้อง |
| `tools/check_licenses.py` | แสดง license ของทุก package ใน venv และ flag GPL/AGPL |

## 1. ติดตั้ง (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
# PyTorch build CUDA 12.6 (GPU NVIDIA); ถ้าไม่มี GPU ให้ใช้ https://download.pytorch.org/whl/cpu
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover tests -v
```

## 2. เก็บภาพ

```powershell
python capture.py --check        # ทดสอบว่ากล้องเปิดได้
python capture.py                # o = OK, n = NG, q = ออก
```

- ภาพเต็มเฟรมจะถูกบันทึกที่ `data/raw/{session}/{OK|NG}/{timestamp}.png` โดย session = วันที่-เวลาที่เริ่มรัน
- **ต้องเก็บอย่างน้อย 3 session** (ระบบแบ่ง train/val/test ทีละ session) ควรเก็บต่างวัน ต่างแสง ต่างล็อตชิ้นงาน และแต่ละ session ต้องมีทั้ง OK และ NG
- ภาพจาก session เดียวกันมักคล้ายกันมาก ถ้าสุ่มแบ่งรายภาพผลจะดีเกินจริง จึงแบ่งตาม session แทน โดยค่าเริ่มต้น session ใหม่สุด = test, ถัดมา = val, ที่เหลือ = train หรือระบุเองได้ที่ `data.val_sessions` / `data.test_sessions`

## 3. ตั้ง ROI และดู augmentation

1. เปิดภาพหนึ่งภาพใน `data/raw/...` แล้ววัดพิกัดกรอบรอบชิ้นงาน จากนั้นตั้ง `preprocess.roi: [x, y, w, h]` ใน `config.yaml` (ควรเป็นสี่เหลี่ยมจัตุรัสเพื่อไม่ให้ภาพเพี้ยนตอน resize) เมื่อรัน `capture.py` ครั้งต่อไปจะเห็นกรอบ ROI บนจอ
2. `python preview_augment.py` จะบันทึกภาพลง `outputs/augment_preview/` โดยช่องซ้ายบนคือภาพต้นฉบับหลัง crop และช่องที่เหลือคือภาพหลัง augment
3. ปรับค่าใน `augment:` จนภาพที่ได้ยังดูเหมือนของจริงที่อาจเจอในไลน์ เช่น ถ้าชิ้นงานวางทิศเดียวเสมอให้ปิด flip/rotate และระวังอย่าให้ augment ลบหรือสร้าง defect ขึ้นมาเอง

## 4. Train

```powershell
python train.py                       # ใช้ train.device: auto (GPU ถ้ามี)
python train.py --device cpu --epochs 20
```

ผลลัพธ์อยู่ที่ `runs/cnn-YYYYMMDD-HHMMSS/` ประกอบด้วย `best.pt` (weights + config ที่ใช้ + การแบ่ง session), `config.yaml`, `history.json`, `loss.png` และ `runs/LATEST` จะชี้ไปที่ run ล่าสุด

## 5. Evaluate และเลือก threshold

```powershell
python evaluate.py --save-threshold
```

- score = ความน่าจะเป็นที่ภาพเป็น NG ถ้า `score >= threshold` จะตัดสินเป็น NG
- threshold **เลือกจาก val split** เป็นค่าสูงสุดที่ยังได้ NG recall ≥ `evaluate.target_ng_recall` (ซึ่งทำให้ false reject ต่ำสุด) แล้วจึง**รายงานผลบน test split** ที่โมเดลไม่เคยเห็น
- false reject rate = จำนวน OK ที่ถูกตัดสินเป็น NG / จำนวน OK ทั้งหมด
- ผลลัพธ์อยู่ที่ `runs/{version}/eval/`: `eval.json`, `confusion_matrix.png`, `threshold_curve.png` ส่วน `--save-threshold` จะเขียน `threshold.json` ให้ server ใช้

## 6. API server

```powershell
python server.py                      # http://127.0.0.1:8000  (เปิดในเบราว์เซอร์เพื่อ upload รูป)
curl http://127.0.0.1:8000/v1/health
curl -F "file=@data/raw/<session>/NG/<file>.png" http://127.0.0.1:8000/v1/inspect
```

```json
{"result": "NG", "score": 0.987654, "threshold": 0.4321, "model_version": "cnn-20261011-201500", "latency_ms": 6.1}
```

- server โหลดโมเดลจาก `inference.checkpoint` หรือจาก `runs/LATEST` ถ้าไม่ได้ระบุ และใช้ threshold จาก `threshold.json` ถ้ามี ไม่มีจะใช้ `inference.threshold`
- preprocessing ใช้ config ที่เก็บอยู่ใน checkpoint เสมอ ภาพที่ส่งเข้ามาต้องเป็นเฟรมเต็มขนาดเดียวกับตอนถ่าย เพราะ ROI ใช้พิกัดบนเฟรมเต็ม
- ค่าเริ่มต้น bind `127.0.0.1` ถ้าต้องให้เครื่องอื่นเรียกได้ให้ใช้ web app ด้านล่างแทนการแก้ `server.host` (API ยังไม่มี authentication)

## 7. Web app (React + Node.js): ถ่ายจากมือถือแล้วส่งมา inference ที่โน๊ตบุ๊ค

```
มือถือ --HTTP LAN--> Node/Express :3000 --localhost--> FastAPI server.py :8000
```

```powershell
python server.py                 # terminal 1 (ที่ root ของโปรเจกต์)
cd web                           # terminal 2 (ต้องมี Node.js 20+)
npm install                      # ครั้งแรก
npm run build                    # build React -> web/client/dist
npm start                        # พิมพ์ URL ของ LAN ออกมา เช่น http://192.168.1.196:3000
```

- เปิด URL ที่ได้จากมือถือที่ต่อ Wi-Fi เดียวกัน ถ้าเปิดไม่ได้ให้อนุญาต Node.js ใน Windows Firewall (Private network) หรือเปิดพอร์ต 3000
- แท็บ **ตรวจ**: ถ่ายรูปแล้วระบบจะส่งไป `/api/inspect` → `/v1/inspect` และแสดงผล OK/NG + score
- แท็บ **เก็บข้อมูล**: ถ่ายรูปแล้วกด OK/NG ระบบจะบันทึกไปที่ `data/raw/{session}/{OK|NG}/*.jpg` ซึ่งใช้ train ได้ทันที ทั้งนี้ควรเปลี่ยน session ทุกครั้งที่เปลี่ยนวัน แสง หรือล็อต
- **ใช้ภาพจากแหล่งเดียวกันทั้งตอน train และตอนตรวจ:** หน้าเว็บจะหมุนภาพตาม EXIF แล้วย่อด้านยาวเหลือ 1280 px ก่อนส่งทุกครั้ง ทำให้ ROI ใน `config.yaml` อิงพิกัดของภาพขนาดนี้ ถ้าจะตรวจด้วยมือถือ ให้เก็บภาพ train ด้วยมือถือผ่านแท็บเก็บข้อมูลด้วย (และถือมือถือแนวเดียวกันทุกครั้ง) เพราะโมเดลที่ train จาก webcam จะทำงานกับภาพมือถือได้ไม่ดี
- บนมือถือผ่าน HTTP จะเปิดแอปกล้องของเครื่องเพื่อถ่าย ส่วน "กล้องสด" (live preview) จะมีให้ใช้เฉพาะเมื่อเปิดจาก `http://localhost:3000` เพราะเบราว์เซอร์อนุญาต `getUserMedia` เฉพาะบน HTTPS หรือ localhost
- ตอนพัฒนาให้ใช้ `npm run dev` ซึ่งรัน Node (`--watch`) คู่กับ Vite ที่ `:5173` แบบ hot reload
- ตั้งค่าผ่าน env ได้: `PORT`, `HOST`, `PY_API` (ค่าเริ่มต้น `http://127.0.0.1:8000`), `RAW_DIR`, `MAX_UPLOAD_MB`
- Node ยังไม่มี authentication จึงควรใช้เฉพาะในเครือข่ายที่ไว้ใจได้

## ทดลองโดยไม่ใช้กล้อง (ข้อมูลจำลอง)

```powershell
python tools/make_synthetic_data.py                        # -> data/synthetic (5 sessions) + data/synthetic/config.yaml
python preview_augment.py --config data/synthetic/config.yaml
python train.py --config data/synthetic/config.yaml        # GPU ~2.5 นาที (CPU ~14 วินาที/epoch)
python evaluate.py --save-threshold
python server.py
```

`make_synthetic_data.py` จะเขียน `data/synthetic/config.yaml` ให้ด้วย โดยเป็นสำเนาของ `config.yaml` ที่ตั้ง ROI รอบชิ้นงานไว้แล้ว ผลที่ได้บนเครื่องนี้ (GTX 1650 Ti): หยุดที่ epoch 75 และ test NG recall 1.00 / false reject 0.00 แต่ข้อมูลจำลองง่ายกว่าของจริงมาก ใช้เพื่อยืนยันว่า pipeline ทำงานได้เท่านั้น

**บทเรียนจากการทดลอง:** ถ้าไม่ตั้ง ROI ภาพเต็มเฟรม 640x480 จะถูกบีบเหลือ 128x128 ทำให้ defect เล็กจนแทบหายไป (รูน็อตเหลือ ~3 px รอยขีดเหลือไม่ถึง 1 px) โมเดลจะจำภาพ train ได้แต่ใช้กับ session อื่นไม่ได้ **ตั้ง ROI ให้ครอบเฉพาะชิ้นงานเสมอ** และอย่าตั้ง `patience` ต่ำ เพราะช่วงแรก val loss แกว่งแรงมาก (เคยกระโดดไปถึง 8 ก่อนจะลดลง)

## Tests

```powershell
python -m unittest discover tests -v
```

## License ของ dependency

### Dependency โดยตรง

| Package | Version | License | ใช้ทำอะไร |
|---|---|---|---|
| torch | 2.14.1 (+cu126) | BSD-3-Clause (wheel รวม third-party ที่เป็น Apache-2.0, MIT, BSL-1.0) | โมเดล / training |
| numpy | 2.4.6 | BSD-3-Clause | array, augmentation |
| opencv-python | 5.0.0.93 | Apache-2.0 (OpenCV), MIT (wrapper) | กล้อง, อ่าน/เขียนภาพ, preprocess, augment |
| fastapi | 0.143.0 | MIT | API server |
| uvicorn | 0.54.0 | BSD-3-Clause | ASGI server |
| python-multipart | 0.0.32 | Apache-2.0 | รับไฟล์ upload |
| scikit-learn | 1.9.1 | BSD-3-Clause | คำนวณ metrics เท่านั้น |
| matplotlib | 3.11.2 | Matplotlib License (แบบ PSF, เข้ากันได้กับ BSD) | กราฟ loss / threshold / confusion matrix |
| PyYAML | 6.0.3 | MIT | อ่าน config.yaml |

### Dependency ที่ลากมาด้วย (ผลจาก `python tools/check_licenses.py`)

| Package | License | | Package | License |
|---|---|---|---|---|
| annotated-doc | MIT | | mpmath | BSD |
| annotated-types | MIT | | narwhals | MIT |
| anyio | MIT | | networkx | BSD-3-Clause |
| click | BSD-3-Clause | | opentelemetry-api | Apache-2.0 |
| cloudpickle | BSD | | packaging | Apache-2.0 OR BSD-2-Clause |
| contourpy | BSD | | pillow | MIT-CMU (HPND) |
| cycler | BSD | | pydantic / pydantic_core | MIT |
| filelock | MIT | | pyparsing | MIT |
| fonttools | MIT | | python-dateutil | Apache-2.0 / BSD |
| fsspec | BSD-3-Clause | | scipy | BSD |
| h11 | MIT | | six | MIT |
| idna | BSD-3-Clause | | starlette | BSD-3-Clause |
| Jinja2 | BSD | | sympy | BSD |
| joblib | BSD-3-Clause | | threadpoolctl | BSD-3-Clause |
| kiwisolver | BSD | | typing-inspection | MIT |
| MarkupSafe | BSD-3-Clause | | typing_extensions | PSF-2.0 |

ผลสแกนล่าสุด: 42 package และไม่พบ GPL/AGPL

### Web app (npm)

| package | license | ใช้ทำอะไร |
|---|---|---|
| express 5 | MIT | Node backend |
| multer 2 | MIT | รับไฟล์ upload |
| react / react-dom 19 | MIT | UI |
| vite 7 / @vitejs/plugin-react 5 | MIT | build (dev เท่านั้น) |

`cd web; npm run check-licenses` จะสแกนทุก package ใน `node_modules` (ผลล่าสุด 136 package, 0 forbidden) ตัวเดียวที่ต้อง review คือ `caniuse-lite` (CC-BY-4.0) ซึ่งเป็นข้อมูลที่ Babel/browserslist ใช้ตอน build และไม่ได้อยู่ในไฟล์ที่ build ออกมา (`npm run check-licenses -- --markdown` จะได้ตารางเต็ม)

### หมายเหตุสำหรับฝ่ายกฎหมายก่อนนำไปใช้ในโรงงาน

- **FFmpeg ใน opencv-python:** wheel บน Windows มี `cv2/opencv_videoio_ffmpeg500_64.dll` (FFmpeg build แบบ LGPL-2.1) มาด้วยเพื่อใช้อ่านไฟล์วิดีโอ โปรเจกต์นี้ไม่ได้ใช้ (กล้องเปิดผ่าน DirectShow) LGPL ไม่ใช่ GPL/AGPL และใช้แบบ dynamic link ได้ แต่ถ้านโยบายห้ามเด็ดขาดควร build OpenCV เองโดยไม่ใส่ FFmpeg
- **License ที่ไม่ใช่ MIT/BSD/Apache ตรงตัวแต่เป็นแบบ permissive:** matplotlib (PSF-based), typing_extensions (PSF-2.0), pillow (MIT-CMU/HPND), BSL-1.0 ใน torch และ Zlib/CC0 ใน numpy
- `PYTHONPATH` บนเครื่องนี้ชี้ไปที่ Spark (`pyspark`, Apache-2.0) ซึ่งไม่ได้เป็น dependency ของโปรเจกต์ และ `check_licenses.py` จะสแกนเฉพาะ package ที่อยู่ใน venv
