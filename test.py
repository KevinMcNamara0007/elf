try:
    import onnxruntime as ort
    print("ONNX Runtime imported successfully!")
except ImportError as e:
    print(f"Error importing ONNX Runtime: {e}")
