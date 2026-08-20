import 'dart:math';
import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../services/api_service.dart';

class ViewfinderHudPainter extends CustomPainter {
  final double scanProgress; // 0.0 to 1.0 (sweeping vertically)
  final double pulseValue;   // 0.0 to 1.0 (pulsing elements)
  final bool isAuditorMode;
  final List<AiDetectionItem> detections;
  final Size? originalImageSize;

  ViewfinderHudPainter({
    required this.scanProgress,
    required this.pulseValue,
    required this.isAuditorMode,
    this.detections = const [],
    this.originalImageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    const double padding = 20.0;
    final Rect targetRect = Rect.fromLTWH(
      padding,
      padding,
      size.width - (padding * 2),
      size.height - (padding * 2),
    );

    final Color accentColor = isAuditorMode ? AppColors.auditorBlue : AppColors.brandPrimary;
    final Color glowColor = isAuditorMode ? AppColors.auditorGlow : AppColors.brandGlow;

    _drawCornerReticles(canvas, targetRect, accentColor);
    _drawLaserScanLine(canvas, targetRect, accentColor, glowColor);
    _drawCenterTargetCrosshair(canvas, size, accentColor);
    _drawCalibrationMarkers(canvas, targetRect, accentColor);

    // Only draw real YOLO detections
    if (detections.isNotEmpty && originalImageSize != null && originalImageSize!.width > 0) {
      _drawRealYoloDetections(canvas, size, originalImageSize!);
    }
  }

  void _drawCornerReticles(Canvas canvas, Rect rect, Color color) {
    const double cornerLen = 28.0;
    const double radius = 12.0;

    final Paint paint = Paint()
      ..color = color
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final Paint dotPaint = Paint()
      ..color = color.withValues(alpha: 0.8 + (pulseValue * 0.2))
      ..style = PaintingStyle.fill;

    // Top-Left
    final Path pathTL = Path()
      ..moveTo(rect.left, rect.top + cornerLen)
      ..lineTo(rect.left, rect.top + radius)
      ..arcToPoint(Offset(rect.left + radius, rect.top), radius: const Radius.circular(radius))
      ..lineTo(rect.left + cornerLen, rect.top);
    canvas.drawPath(pathTL, paint);
    canvas.drawCircle(Offset(rect.left + 5, rect.top + 5), 2.0, dotPaint);

    // Top-Right
    final Path pathTR = Path()
      ..moveTo(rect.right - cornerLen, rect.top)
      ..lineTo(rect.right - radius, rect.top)
      ..arcToPoint(Offset(rect.right, rect.top + radius), radius: const Radius.circular(radius))
      ..lineTo(rect.right, rect.top + cornerLen);
    canvas.drawPath(pathTR, paint);
    canvas.drawCircle(Offset(rect.right - 5, rect.top + 5), 2.0, dotPaint);

    // Bottom-Left
    final Path pathBL = Path()
      ..moveTo(rect.left, rect.bottom - cornerLen)
      ..lineTo(rect.left, rect.bottom - radius)
      ..arcToPoint(Offset(rect.left + radius, rect.bottom), radius: const Radius.circular(radius))
      ..lineTo(rect.left + cornerLen, rect.bottom);
    canvas.drawPath(pathBL, paint);
    canvas.drawCircle(Offset(rect.left + 5, rect.bottom - 5), 2.0, dotPaint);

    // Bottom-Right
    final Path pathBR = Path()
      ..moveTo(rect.right - cornerLen, rect.bottom)
      ..lineTo(rect.right - radius, rect.bottom)
      ..arcToPoint(Offset(rect.right, rect.bottom - radius), radius: const Radius.circular(radius))
      ..lineTo(rect.right, rect.bottom - cornerLen);
    canvas.drawPath(pathBR, paint);
    canvas.drawCircle(Offset(rect.right - 5, rect.bottom - 5), 2.0, dotPaint);
  }

  void _drawLaserScanLine(Canvas canvas, Rect rect, Color accentColor, Color glowColor) {
    final double scanY = rect.top + (rect.height * scanProgress);

    // 1. Glow Beam (Gradient tail)
    const double tailHeight = 35.0;
    final Rect tailRect = Rect.fromLTRB(rect.left, max(rect.top, scanY - tailHeight), rect.right, scanY);

    final Paint tailPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          accentColor.withValues(alpha: 0.0),
          accentColor.withValues(alpha: 0.18 + (pulseValue * 0.08)),
        ],
      ).createShader(tailRect);
    canvas.drawRect(tailRect, tailPaint);

    // 2. High-intensity Laser Line
    final Paint laserLine = Paint()
      ..shader = LinearGradient(
        colors: [
          accentColor.withValues(alpha: 0.0),
          accentColor.withValues(alpha: 0.9),
          Colors.white,
          accentColor.withValues(alpha: 0.9),
          accentColor.withValues(alpha: 0.0),
        ],
        stops: const [0.0, 0.25, 0.5, 0.75, 1.0],
      ).createShader(Rect.fromLTWH(rect.left, scanY - 1.5, rect.width, 3))
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    canvas.drawLine(
      Offset(rect.left, scanY),
      Offset(rect.right, scanY),
      laserLine,
    );

    // 3. Center laser beacon point
    final Paint centerDot = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(rect.center.dx, scanY), 2.5, centerDot);
  }

  void _drawCenterTargetCrosshair(Canvas canvas, Size size, Color color) {
    final Offset center = Offset(size.width / 2, size.height / 2);
    const double crosshairSize = 14.0;

    final Paint gridPaint = Paint()
      ..color = color.withValues(alpha: 0.35)
      ..strokeWidth = 1.2
      ..style = PaintingStyle.stroke;

    // Crosshair Lines
    canvas.drawLine(
      Offset(center.dx - crosshairSize, center.dy),
      Offset(center.dx + crosshairSize, center.dy),
      gridPaint,
    );
    canvas.drawLine(
      Offset(center.dx, center.dy - crosshairSize),
      Offset(center.dx, center.dy + crosshairSize),
      gridPaint,
    );

    // Outer Target Ring
    final double radius = 32.0 + (pulseValue * 4.0);
    final Paint ringPaint = Paint()
      ..color = color.withValues(alpha: 0.18 + (pulseValue * 0.12))
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(center, radius, ringPaint);
  }

  void _drawCalibrationMarkers(Canvas canvas, Rect rect, Color color) {
    final Paint tickPaint = Paint()
      ..color = color.withValues(alpha: 0.4)
      ..strokeWidth = 1.0;

    const int tickCount = 6;
    final double spacing = rect.width / (tickCount + 1);

    for (int i = 1; i <= tickCount; i++) {
      final double x = rect.left + (spacing * i);
      canvas.drawLine(Offset(x, rect.top), Offset(x, rect.top + 4), tickPaint);
      canvas.drawLine(Offset(x, rect.bottom), Offset(x, rect.bottom - 4), tickPaint);
    }
  }

  void _drawRealYoloDetections(Canvas canvas, Size canvasSize, Size imgSize) {
    if (imgSize.width <= 0 || imgSize.height <= 0) return;

    final double imgAspect = imgSize.width / imgSize.height;
    final double canvasAspect = canvasSize.width / canvasSize.height;
    double renderW, renderH, offsetX, offsetY;

    if (canvasAspect > imgAspect) {
      renderH = canvasSize.height;
      renderW = renderH * imgAspect;
      offsetX = (canvasSize.width - renderW) / 2.0;
      offsetY = 0.0;
    } else {
      renderW = canvasSize.width;
      renderH = renderW / imgAspect;
      offsetX = 0.0;
      offsetY = (canvasSize.height - renderH) / 2.0;
    }

    final double scale = renderW / imgSize.width;

    for (final item in detections) {
      if (item.box.length < 4) continue;

      final double x1 = offsetX + (item.box[0] * scale);
      final double y1 = offsetY + (item.box[1] * scale);
      final double x2 = offsetX + (item.box[2] * scale);
      final double y2 = offsetY + (item.box[3] * scale);
      final Rect boxRect = Rect.fromLTRB(x1, y1, x2, y2);

      Color tierColor;
      switch (item.className.toLowerCase()) {
        case 'fresh':
        case 'ripe':
          tierColor = AppColors.tierFresh;
          break;
        case 'unripe':
          tierColor = AppColors.tierUnripe;
          break;
        case 'overripe':
          tierColor = AppColors.tierOverripe;
          break;
        case 'spoiled':
        default:
          tierColor = AppColors.tierSpoiled;
          break;
      }

      // Box Background & Border
      final Paint boxPaint = Paint()
        ..color = tierColor.withValues(alpha: 0.85)
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke;

      final Paint bgPaint = Paint()
        ..color = tierColor.withValues(alpha: 0.12)
        ..style = PaintingStyle.fill;

      canvas.drawRRect(RRect.fromRectAndRadius(boxRect, const Radius.circular(8)), bgPaint);
      canvas.drawRRect(RRect.fromRectAndRadius(boxRect, const Radius.circular(8)), boxPaint);

      // Corner Brackets
      const double markLen = 8.0;
      final Paint cornerPaint = Paint()
        ..color = Colors.white
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke;

      canvas.drawLine(Offset(boxRect.left, boxRect.top + markLen), Offset(boxRect.left, boxRect.top), cornerPaint);
      canvas.drawLine(Offset(boxRect.left, boxRect.top), Offset(boxRect.left + markLen, boxRect.top), cornerPaint);
      canvas.drawLine(Offset(boxRect.right - markLen, boxRect.bottom), Offset(boxRect.right, boxRect.bottom), cornerPaint);
      canvas.drawLine(Offset(boxRect.right, boxRect.bottom - markLen), Offset(boxRect.right, boxRect.bottom), cornerPaint);

      // Label Tag
      String displayClass = item.className.toLowerCase();
      if (displayClass == 'fresh') displayClass = 'ripe';
      final String labelText = '${displayClass.toUpperCase()} ${(item.confidence * 100).toStringAsFixed(0)}%';
      final textSpan = TextSpan(
        text: labelText,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 10.0,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.5,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final tagBg = Rect.fromLTWH(
        boxRect.left,
        max(4.0, boxRect.top - 16),
        textPainter.width + 8,
        15,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(tagBg, const Radius.circular(4)),
        Paint()..color = tierColor,
      );
      textPainter.paint(canvas, Offset(tagBg.left + 4, tagBg.top + 1));
    }
  }

  @override
  bool shouldRepaint(covariant ViewfinderHudPainter oldDelegate) {
    return oldDelegate.scanProgress != scanProgress ||
        oldDelegate.pulseValue != pulseValue ||
        oldDelegate.isAuditorMode != isAuditorMode ||
        oldDelegate.detections != detections;
  }
}

