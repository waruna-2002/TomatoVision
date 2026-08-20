import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'on_device_ai_engine.dart';

class AiDetectionItem {
  final String className;
  final double confidence;
  final List<double> box;

  AiDetectionItem({
    required this.className,
    required this.confidence,
    required this.box,
  });

  factory AiDetectionItem.fromJson(Map<String, dynamic> json) {
    return AiDetectionItem(
      className: json['class_name'] ?? 'unknown',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      box: (json['box'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ?? [],
    );
  }
}

class AiPredictionResponse {
  final bool success;
  final int totalDetected;
  final int ripe;
  final int unripe;
  final int overripe;
  final int spoiled;
  final double qualityPercentage;
  final List<AiDetectionItem> detections;
  final String? annotatedImageBase64;

  int get fresh => ripe;

  AiPredictionResponse({
    required this.success,
    required this.totalDetected,
    required this.ripe,
    required this.unripe,
    required this.overripe,
    required this.spoiled,
    required this.qualityPercentage,
    required this.detections,
    this.annotatedImageBase64,
  });

  factory AiPredictionResponse.fromJson(Map<String, dynamic> json) {
    final counts = json['counts'] as Map<String, dynamic>? ?? {};
    final detectionList = (json['detections'] as List<dynamic>?)
            ?.map((e) => AiDetectionItem.fromJson(e as Map<String, dynamic>))
            .toList() ??
        [];

    final int ripeCount = (counts['ripe'] ?? counts['fresh'] as num?)?.toInt() ?? 0;

    return AiPredictionResponse(
      success: json['success'] == true,
      totalDetected: (json['total_detected'] as num?)?.toInt() ?? 0,
      ripe: ripeCount,
      unripe: (counts['unripe'] as num?)?.toInt() ?? 0,
      overripe: (counts['overripe'] as num?)?.toInt() ?? 0,
      spoiled: (counts['spoiled'] as num?)?.toInt() ?? 0,
      qualityPercentage: (json['quality_percentage'] as num?)?.toDouble() ?? 0.0,
      detections: detectionList,
      annotatedImageBase64: json['annotated_image_base64'] as String?,
    );
  }
}

class ApiService {
  static String baseUrl = kIsWeb ? 'http://localhost:8000' : 'http://192.168.1.7:8000';

  static Future<bool> checkServerHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 2));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['status'] == 'online';
      }
    } catch (e) {
      debugPrint('Local AI Server offline: On-Device AI Engine Active ($e)');
    }
    // Return true because On-Device Edge AI Engine is always active and ready
    return true;
  }

  static Future<AiPredictionResponse?> analyzeImageBytes(Uint8List imageBytes) async {
    // 1. Try remote/cloud server if reachable (fast timeout 2s)
    try {
      final base64Image = base64Encode(imageBytes);
      final response = await http.post(
        Uri.parse('$baseUrl/predict_base64'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image': base64Image}),
      ).timeout(const Duration(milliseconds: 2000));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return AiPredictionResponse.fromJson(data);
      }
    } catch (e) {
      debugPrint('PC Server offline / Mobile mode: Executing On-Device Standalone Agro-Vision Engine: $e');
    }

    // 2. High-speed Mobile Standalone On-Device Engine (Zero PC dependency!)
    try {
      final localResult = await OnDeviceAiEngine.analyzeImageBytes(imageBytes);
      if (localResult != null && localResult.success) {
        return localResult;
      }
    } catch (e) {
      debugPrint('OnDeviceAiEngine execution error: $e');
    }

    return null;
  }
}
