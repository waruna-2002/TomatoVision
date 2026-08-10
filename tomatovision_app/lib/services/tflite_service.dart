import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart';

class TfliteService {
  Interpreter? _interpreter;
  bool _isModelLoaded = false;

  bool get isModelLoaded => _isModelLoaded;

  Future<void> loadModel() async {
    // Web එකේදී C++ Native FFI වැඩ නොකරන නිසා Web Check එකක් යෙදීම:
    if (kIsWeb) {
      debugPrint('Running on Web: TFLite Native model skipped (Mock Mode Active)');
      _isModelLoaded = true;
      return;
    }

    try {
      _interpreter = await Interpreter.fromAsset('assets/models/tomato_model.tflite');
      _isModelLoaded = true;
      debugPrint('TFLite Model Loaded Successfully on Native Device');
    } catch (e) {
      debugPrint('Error loading TFLite model: $e');
    }
  }

  void close() {
    _interpreter?.close();
  }
}