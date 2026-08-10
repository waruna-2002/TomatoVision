abstract class TfliteService {
  factory TfliteService() => throw UnsupportedError('TFLite is not supported on Web');
  Future<void> loadModel();
  void close();
  bool get isModelLoaded;
}

class TfliteHelper implements TfliteService {
  bool _isLoaded = false;

  @override
  bool get isModelLoaded => _isLoaded;

  @override
  Future<void> loadModel() async {
    print('Running on Web: TFLite Native model skipped (Mock Mode Active)');
    _isLoaded = true;
  }

  @override
  void close() {}
}

TfliteService getTfliteService() => TfliteHelper();