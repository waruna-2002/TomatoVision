import '../core/utils/score_calculator.dart';

class ScanRecord {
  final String id;
  final DateTime timestamp;
  final QualityResult result;
  final int unripe;
  final int fresh;
  final int overripe;
  final int spoiled;
  final bool isAuditorMode;

  ScanRecord({
    required this.id,
    required this.timestamp,
    required this.result,
    required this.unripe,
    required this.fresh,
    required this.overripe,
    required this.spoiled,
    required this.isAuditorMode,
  });

  int get totalCount => unripe + fresh + overripe + spoiled;
}

class HistoryService {
  static final List<ScanRecord> _records = [
    ScanRecord(
      id: 'TV-98214',
      timestamp: DateTime.now().subtract(const Duration(minutes: 14)),
      result: ScoreCalculator.calculate(unripeCount: 1, freshCount: 9, overripeCount: 2, spoiledCount: 0),
      unripe: 1,
      fresh: 9,
      overripe: 2,
      spoiled: 0,
      isAuditorMode: true,
    ),
    ScanRecord(
      id: 'TV-87102',
      timestamp: DateTime.now().subtract(const Duration(hours: 1, minutes: 22)),
      result: ScoreCalculator.calculate(unripeCount: 3, freshCount: 5, overripeCount: 1, spoiledCount: 1),
      unripe: 3,
      fresh: 5,
      overripe: 1,
      spoiled: 1,
      isAuditorMode: false,
    ),
  ];

  static List<ScanRecord> get records => List.unmodifiable(_records);

  static void addRecord(ScanRecord record) {
    _records.insert(0, record);
    if (_records.length > 50) {
      _records.removeLast();
    }
  }

  static void clearHistory() {
    _records.clear();
  }
}
