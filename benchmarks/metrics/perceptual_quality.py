"""Perceptual quality metrics (PESQ, STOI, UTMOS, NISQA)."""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_pesq(
    reference_audio: np.ndarray,
    degraded_audio: np.ndarray,
    sample_rate: int = 16000
) -> float:
    """Calculate PESQ (Perceptual Evaluation of Speech Quality).
    
    PESQ is an ITU standard for objective speech quality assessment.
    Range: -0.5 to 4.5 (higher is better)
    
    Args:
        reference_audio: Original/reference audio
        degraded_audio: Degraded/reconstructed audio
        sample_rate: Audio sample rate (8000 or 16000 Hz)
        
    Returns:
        PESQ score
    """
    try:
        from pesq import pesq
        
        # PESQ requires same length
        min_len = min(len(reference_audio), len(degraded_audio))
        reference_audio = reference_audio[:min_len]
        degraded_audio = degraded_audio[:min_len]
        
        # PESQ only supports 8kHz or 16kHz
        if sample_rate not in [8000, 16000]:
            logger.warning(f"PESQ requires 8kHz or 16kHz, got {sample_rate}Hz. Resampling...")
            import librosa
            reference_audio = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=16000)
            degraded_audio = librosa.resample(degraded_audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        mode = 'wb' if sample_rate == 16000 else 'nb'
        score = pesq(sample_rate, reference_audio, degraded_audio, mode)
        
        logger.info(f"PESQ score: {score:.3f}")
        return float(score)
        
    except ImportError:
        logger.error("pesq package not installed. Install with: pip install pesq")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating PESQ: {e}")
        return 0.0


def calculate_stoi(
    reference_audio: np.ndarray,
    degraded_audio: np.ndarray,
    sample_rate: int = 16000
) -> float:
    """Calculate STOI (Short-Time Objective Intelligibility).
    
    STOI measures speech intelligibility.
    Range: 0.0 to 1.0 (higher is better)
    
    Args:
        reference_audio: Original/reference audio
        degraded_audio: Degraded/reconstructed audio
        sample_rate: Audio sample rate
        
    Returns:
        STOI score
    """
    try:
        from pystoi import stoi
        
        # STOI requires same length
        min_len = min(len(reference_audio), len(degraded_audio))
        reference_audio = reference_audio[:min_len]
        degraded_audio = degraded_audio[:min_len]
        
        # STOI typically uses 10kHz, but works with other rates
        score = stoi(reference_audio, degraded_audio, sample_rate, extended=False)
        
        logger.info(f"STOI score: {score:.3f}")
        return float(score)
        
    except ImportError:
        logger.error("pystoi package not installed. Install with: pip install pystoi")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating STOI: {e}")
        return 0.0


def calculate_utmos(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> float:
    """Calculate UTMOS (Universal Text-to-Speech Mean Opinion Score).
    
    UTMOS predicts subjective quality (MOS) using a pretrained model.
    Range: 1.0 to 5.0 (higher is better)
    
    Args:
        audio: Audio signal to evaluate
        sample_rate: Audio sample rate
        
    Returns:
        Predicted MOS score
    """
    try:
        import tempfile
        import os
        import torch
        import utmosv2
        from scipy.io import wavfile
        
        logger.info("Loading UTMOS model...")
        
        # Force CPU usage if CUDA is not available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA for UTMOS")
        else:
            logger.info("CUDA not available, using CPU for UTMOS")
        
        model = utmosv2.create_model(pretrained=True, device=device)
        
        # Resample if needed (UTMOS expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Convert to int16 for wav file
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save to temporary file (UTMOS expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            wavfile.write(tmp_path, sample_rate, audio_int16)
        
        # Predict
        score = model.predict(input_path=tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Handle both single value and list return types
        if isinstance(score, list):
            # If multiple predictions, take the first one
            score = score[0]['score'] if isinstance(score[0], dict) else score[0]
        
        # UTMOS outputs range is typically 1-5
        logger.info(f"UTMOS score: {score:.3f}")
        return float(score)
        
    except ImportError:
        logger.error("utmosv2 not installed. Install with: pip install git+https://github.com/sarulab-speech/UTMOSv2.git")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating UTMOS: {e}")
        return 0.0


def calculate_nisqa(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> dict:
    """Calculate NISQA (Non-Intrusive Speech Quality Assessment).
    
    NISQA predicts multiple quality dimensions without reference.
    
    Args:
        audio: Audio signal to evaluate
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary with NISQA scores
    """
    try:
        import torch
        import os
        from nisqa.NISQA_model import nisqaModel
        
        logger.info("Loading NISQA model...")
        
        # Resample if needed (NISQA expects 48kHz)
        if sample_rate != 48000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=48000)
            sample_rate = 48000
        
        # Save audio to temporary file (NISQA expects file path)
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name
        
        try:
            # Prepare arguments for NISQA model
            # NISQA_DIM predicts MOS + 4 quality dimensions (noisiness, coloration, discontinuity, loudness)
            args = {
                'mode': 'predict_file',
                'pretrained_model': 'weights/nisqa.tar',
                'deg': tmp_path,
                'data_dir': None,
                'output_dir': None,
                'model': 'NISQA_DIM',
                # Mel-spectrogram parameters (required by NISQA)
                'ms_channel': None,  # Use all channels
                # Don't specify ms_win_length - let NISQA use its default
                # (The parameter is in milliseconds and gets converted internally)
                # Training/inference parameters
                'tr_bs_val': 1,
                'tr_num_workers': 0,
                'tr_parallel': False,
            }
            
            # Initialize and run model
            model = nisqaModel(args)
            results_df = model.predict()
            
            # Extract results from dataframe
            if len(results_df) > 0:
                row = results_df.iloc[0]
                results = {
                    'mos': float(row.get('mos_pred', 0.0)),
                    'noisiness': float(row.get('noi_pred', 0.0)),
                    'coloration': float(row.get('col_pred', 0.0)),
                    'discontinuity': float(row.get('dis_pred', 0.0)),
                    'loudness': float(row.get('loud_pred', 0.0))
                }
                
                logger.info(f"NISQA MOS: {results['mos']:.3f}")
                if results['noisiness'] > 0:
                    logger.info(f"NISQA Noisiness: {results['noisiness']:.3f}")
                    logger.info(f"NISQA Coloration: {results['coloration']:.3f}")
                    logger.info(f"NISQA Discontinuity: {results['discontinuity']:.3f}")
                    logger.info(f"NISQA Loudness: {results['loudness']:.3f}")
                
                return results
            else:
                logger.error("NISQA returned empty results")
                return {'mos': 0.0}
                
        finally:
            os.unlink(tmp_path)
        
    except ImportError as e:
        logger.error(f"NISQA not installed: {e}")
        logger.error("Install from: https://github.com/gabrielmittag/NISQA")
        logger.error("Or: pip install nisqa")
        return {'mos': 0.0}
    except FileNotFoundError as e:
        logger.error(f"NISQA model weights not found: {e}")
        logger.error("Make sure NISQA is properly installed with model weights")
        return {'mos': 0.0}
    except Exception as e:
        logger.error(f"Error calculating NISQA: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'mos': 0.0}


def calculate_all_perceptual_metrics(
    reference_audio: np.ndarray,
    degraded_audio: np.ndarray,
    sample_rate: int = 16000
) -> dict:
    """Calculate all perceptual quality metrics.
    
    Args:
        reference_audio: Original audio
        degraded_audio: Reconstructed audio
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary with all metric scores
    """
    results = {}
    
    # Intrusive metrics (need reference)
    results['pesq'] = calculate_pesq(reference_audio, degraded_audio, sample_rate)
    results['stoi'] = calculate_stoi(reference_audio, degraded_audio, sample_rate)
    
    # Non-intrusive metrics (only degraded audio)
    results['utmos'] = calculate_utmos(degraded_audio, sample_rate)
    results['nisqa'] = calculate_nisqa(degraded_audio, sample_rate)
    
    return results
