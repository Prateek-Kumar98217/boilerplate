# 02 — Data Preprocessing

Modular preprocessing pipelines for text, audio, images, video, and tabular ML data.

## Structure

```
02-data-preprocessing/
├── config.py                     # Pydantic Settings for all modalities
├── .env.example                  # Environment template
├── text/
│   ├── cleaner.py                # Unicode norm, URL/HTML strip, lemmatize, stop-words
│   └── tokenizer.py              # HuggingFace AutoTokenizer wrapper + sliding-window
├── audio/
│   └── processor.py              # Librosa / torchaudio: mel spec, MFCC, Whisper input
├── images/
│   └── processor.py              # torchvision transforms: train/val/inference pipelines
├── video/
│   └── processor.py              # OpenCV frame extraction + temporal sampling
└── ml/
    ├── analysis.py               # EDA: stats, missing, outliers, correlations, MI
    ├── cleaning.py               # Imputation, outlier removal, stratified split
    ├── feature_engineering.py    # Scaling, encoding, poly features, PCA, SelectKBest
    └── examples/
        └── ml_pipeline.py        # Full tabular preprocessing example
```

## Quick Examples

### Text

```python
from text.cleaner import TextCleaner
from text.tokenizer import HFTokenizer

cleaner = TextCleaner(remove_stopwords=True, lemmatize=True)
clean_text = cleaner("Check out https://example.com — it's incredible!!")

tok = HFTokenizer("bert-base-uncased", max_length=128)
enc = tok(["Hello world", "AI/ML is great"])
# enc["input_ids"], enc["attention_mask"]
```

### Audio

```python
from audio.processor import AudioProcessor

proc = AudioProcessor(sample_rate=16000, n_mels=80)
result = proc.process_file("audio.wav")
# result["waveform"], result["mel_spectrogram"], result["mfcc"]
```

### Images

```python
from images.processor import ImageProcessor

proc = ImageProcessor(size=224)
train_tfm = proc.get_train_transforms()
val_tfm   = proc.get_val_transforms()

img = proc.load("photo.jpg")
tensor = train_tfm(img)   # → (3, 224, 224)
```

### Video

```python
from video.processor import VideoProcessor

proc = VideoProcessor(target_fps=1, max_frames=32, resize=(224, 224))
frames = proc.extract_frames("clip.mp4", sampling="uniform")
tensor = proc.to_tensor(frames)   # → (T, 3, 224, 224)
```

### ML Tabular

```python
from ml.analysis import DataAnalyzer
from ml.cleaning import DataCleaner, stratified_split
from ml.feature_engineering import FeatureEngineer

analyzer = DataAnalyzer(df, target_col="label")
print(analyzer.missing_summary())
print(analyzer.feature_importance())

df_train, df_val, df_test = stratified_split(df, "label")
cleaner = DataCleaner(missing_strategy="median", outlier_method="iqr")
df_train_clean = cleaner.fit_transform(df_train)
df_val_clean   = cleaner.transform(df_val)

fe = FeatureEngineer(scaler="standard", pca_components=50)
X_train = fe.fit_transform(df_train_clean, numeric_cols)
X_val   = fe.transform(df_val_clean)
```
