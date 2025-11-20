"""Quick test script for benchmarking system."""

import numpy as np
from benchmarks.metrics.wer import calculate_wer
from benchmarks.metrics.speaker_similarity import cosine_similarity


def test_wer():
    """Test WER calculation."""
    print("Testing WER calculation...")
    
    # Perfect match
    ref = "hello world"
    hyp = "hello world"
    wer = calculate_wer(ref, hyp)
    print(f"  Perfect match: WER = {wer:.3f} (expected 0.000)")
    assert wer == 0.0
    
    # One substitution
    ref = "hello world"
    hyp = "hello word"
    wer = calculate_wer(ref, hyp)
    print(f"  One substitution: WER = {wer:.3f} (expected 0.500)")
    assert abs(wer - 0.5) < 0.01
    
    # One deletion
    ref = "hello world"
    hyp = "hello"
    wer = calculate_wer(ref, hyp)
    print(f"  One deletion: WER = {wer:.3f} (expected 0.500)")
    assert abs(wer - 0.5) < 0.01
    
    print("✅ WER tests passed!\n")


def test_speaker_similarity():
    """Test speaker similarity calculation."""
    print("Testing speaker similarity...")
    
    # Identical vectors
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.array([1, 2, 3, 4, 5], dtype=float)
    sim = cosine_similarity(a, b)
    print(f"  Identical vectors: similarity = {sim:.3f} (expected 1.000)")
    assert abs(sim - 1.0) < 0.01
    
    # Orthogonal vectors
    a = np.array([1, 0], dtype=float)
    b = np.array([0, 1], dtype=float)
    sim = cosine_similarity(a, b)
    print(f"  Orthogonal vectors: similarity = {sim:.3f} (expected 0.000)")
    assert abs(sim) < 0.01
    
    # Opposite vectors
    a = np.array([1, 2, 3], dtype=float)
    b = np.array([-1, -2, -3], dtype=float)
    sim = cosine_similarity(a, b)
    print(f"  Opposite vectors: similarity = {sim:.3f} (expected -1.000)")
    assert abs(sim + 1.0) < 0.01
    
    print("✅ Speaker similarity tests passed!\n")


def test_import():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from benchmarks.metrics import wer
        print("  ✅ wer module")
    except ImportError as e:
        print(f"  ❌ wer module: {e}")
    
    try:
        from benchmarks.metrics import speaker_similarity
        print("  ✅ speaker_similarity module")
    except ImportError as e:
        print(f"  ❌ speaker_similarity module: {e}")
    
    try:
        from benchmarks.metrics import perceptual_quality
        print("  ✅ perceptual_quality module")
    except ImportError as e:
        print(f"  ❌ perceptual_quality module: {e}")
    
    try:
        from benchmarks.baselines import opus
        print("  ✅ opus baseline")
    except ImportError as e:
        print(f"  ❌ opus baseline: {e}")
    
    try:
        from benchmarks.baselines import encodec
        print("  ✅ encodec baseline")
    except ImportError as e:
        print(f"  ❌ encodec baseline: {e}")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARKING SYSTEM TESTS")
    print("=" * 60)
    print()
    
    test_import()
    test_wer()
    test_speaker_similarity()
    
    print("=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Install additional dependencies:")
    print("   uv pip install encodec scikit-learn")
    print()
    print("2. Run full benchmark:")
    print("   uv run python -m benchmarks.run_all \\")
    print("       --config minimal_mode \\")
    print("       --audio test_audios/tpo53-1.wav")
