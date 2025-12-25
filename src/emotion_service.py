import logging
from io import BytesIO
import base64

import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel

logger = logging.getLogger(__name__)


class VoiceEmotionDetector:
    """
    Ultra-fast GPU-optimized emotion detection for single inference.
    Optimized for real-time chat applications.
    """
    
    def __init__(self, production_model_path: str = "production_model.pt"):
        """Initialize optimized emotion detector."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        
        logger.info(f"| {self.device.upper() if self.device == 'cuda' else 'CPU '} SERVICE                        |       DONE          |\n|-----------------------------------------------------------|")
        
        # Load models
        MODEL_NAME = "facebook/hubert-base-ls960"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        self.hubert_model = HubertModel.from_pretrained(MODEL_NAME).to(self.device)
        self.hubert_model.eval()
        
        # Load production model

        self.production_model = torch.jit.load(production_model_path, map_location=self.device)
        self.production_model.eval()
        
        # Apply torch.compile if available (PyTorch 2.0+)
        self._optimize_models()
        
        # Emotion mapping
        self.emotion_map = {0: "Positive", 1: "Neutral", 2: "Negative"}
        
        # Enable mixed precision for 2x speedup
        self.use_amp = torch.cuda.is_available()
        
        # Warm up GPU
        self._warmup()
    
    def _optimize_models(self):
        """Apply torch.compile for maximum speed."""

        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        try:
            if hasattr(torch, 'compile'):
                self.hubert_model = torch.compile(
                    self.hubert_model, 
                    mode='reduce-overhead',
                    fullgraph=True
                )
                self.production_model = torch.compile(
                    self.production_model,
                    mode='reduce-overhead'
                )
                logger.info("| MODEL COMPILATION                   |       DONE          |\n|-----------------------------------------------------------|")

        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    def _warmup(self):
        """Warm up GPU with dummy inference to avoid cold start."""
        try:
            dummy_audio = torch.randn(1, 16000)
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        inputs = self.processor(
                            dummy_audio.squeeze(), 
                            sampling_rate=16000, 
                            return_tensors="pt"
                        ).to(self.device)
                        output = self.hubert_model(**inputs).last_hidden_state
                        pooled = torch.cat([output.mean(dim=1), output.std(dim=1)], dim=1)
                        _ = self.production_model(pooled)
                else:
                    inputs = self.processor(
                        dummy_audio.squeeze(), 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    ).to(self.device)
                    output = self.hubert_model(**inputs).last_hidden_state
                    pooled = torch.cat([output.mean(dim=1), output.std(dim=1)], dim=1)
                    _ = self.production_model(pooled)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("| GPU WARMUP                          |       DONE          |\n|-----------------------------------------------------------|")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def _load_audio_from_base64(self, audio_base64: str, audio_format: str, 
                                 target_sr: int = 16000, max_duration: float = 10.0) -> torch.Tensor:
        """
        Load and preprocess audio from base64 string.
        
        Args:
            audio_base64: Base64 encoded audio
            audio_format: Audio format (webm, mp3, wav, etc.)
            target_sr: Target sample rate
            max_duration: Maximum duration in seconds
            
        Returns:
            Preprocessed waveform tensor
        """
        try:
            # Decode base64
            if audio_base64.startswith('data:'):
                audio_base64 = audio_base64.split(',')[1]
            audio_bytes = base64.b64decode(audio_base64)
            audio_io = BytesIO(audio_bytes)
            
            # Load audio
            try:
                waveform, sr = torchaudio.load(audio_io)
            except Exception as e:
                logger.warning(f"torchaudio failed: {e}, using soundfile")
                audio_io.seek(0)
                data, sr = sf.read(audio_io, dtype="float32")
                if data.ndim == 1:
                    data = data[None, :]
                else:
                    data = data.T
                waveform = torch.from_numpy(data)
            
            # Resample if needed
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Trim to max duration
            max_len = int(target_sr * max_duration)
            waveform = waveform[:, :max_len]
            
            # Check minimum length
            if waveform.shape[-1] < 1600:  # < 0.1s
                raise ValueError("Audio too short (< 0.1s)")
            
            return waveform
            
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            raise ValueError(f"Audio loading failed: {str(e)}")
    
    def _extract_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract HuBERT embeddings from waveform.
        
        Args:
            waveform: Audio waveform tensor
            
        Returns:
            Pooled embedding tensor
        """
        inputs = self.processor(
            waveform.squeeze(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    output = self.hubert_model(**inputs).last_hidden_state
                    pooled = torch.cat([
                        output.mean(dim=1),
                        output.std(dim=1)
                    ], dim=1)
            else:
                output = self.hubert_model(**inputs).last_hidden_state
                pooled = torch.cat([
                    output.mean(dim=1),
                    output.std(dim=1)
                ], dim=1)
        
        return pooled
    
    def detect_emotion(self, audio_base64: str, audio_format: str = "webm") -> dict:
        """
        Detect emotion from base64 audio (SYNCHRONOUS - optimized for speed).
        
        Args:
            audio_base64: Base64 encoded audio string
            audio_format: Audio format (default: "webm")
        
        Returns:
            dict with:
                - 'label' (int): 0=Positive, 1=Neutral, 2=Negative
                - 'emotion' (str): Emotion name
                - 'confidence' (float): Prediction confidence
        """
        try:
            # Load and preprocess audio
            waveform = self._load_audio_from_base64(audio_base64, audio_format)
            
            # Extract embeddings
            embeddings = self._extract_embeddings(waveform)
            
            # Predict emotion
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        prediction = self.production_model(embeddings)
                else:
                    prediction = self.production_model(embeddings)
                
                # Get probabilities
                if prediction.shape[-1] > 1:
                    probabilities = torch.softmax(prediction, dim=-1)
                    confidence = torch.max(probabilities).item()
                    label = torch.argmax(prediction, dim=-1).item()
                else:
                    label = torch.argmax(prediction).item()
                    confidence = None
            
            emotion = self.emotion_map[label]
            
            result = {
                "label": label,
                "emotion": emotion,
                "confidence": confidence
            }
            
            # ✅ FIXED: Proper conditional formatting
            confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
            logger.info(f"..........Emotion: {emotion} (confidence: {confidence_str})..........🍀")
            
            return result
            
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            raise ValueError(f"Emotion detection failed: {str(e)}")
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("| GPU CLEANUP                         |       DONE          |\n|-----------------------------------------------------------|")
