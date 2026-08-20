import 'tflite_stub.dart';

class TfliteMobile implements TfliteService {
  bool _isLoaded = false;

  @override
  bool get isModelLoaded => _isLoaded;

  @override
  Future<void> loadModel() async {
    _isLoaded = true;
    print('TomatoVision Mobile Engine Ready');
  }

  @override
  void close() {}
}

TfliteService getTfliteService() => TfliteMobile();