import 'package:tflite_flutter/tflite_flutter.dart';
import 'tflite_stub.dart';

class TfliteMobile implements TfliteService {
  Interpreter? _interpreter;
  bool _isLoaded = false;

  @override
  bool get isModelLoaded => _isLoaded;

  @override
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/tomato_model.tflite');
      _isLoaded = true;
      print('TFLite Model Loaded Successfully');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  @override
  void close() {
    _interpreter?.close();
  }
}

TfliteService getTfliteService() => TfliteMobile();