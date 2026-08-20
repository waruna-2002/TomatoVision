import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

class QualityResult {
  final double scorePercentage;
  final String gradeLetter;
  final String recommendationBadge;
  final String aiInsight;
  final String shelfLifeDays;
  final String optimalTemp;
  final String optimalHumidity;
  final String riskAssessment;
  final Color badgeColor;
  final List<Color> gradientColors;

  QualityResult({
    required this.scorePercentage,
    required this.gradeLetter,
    required this.recommendationBadge,
    required this.aiInsight,
    required this.shelfLifeDays,
    required this.optimalTemp,
    required this.optimalHumidity,
    required this.riskAssessment,
    required this.badgeColor,
    required this.gradientColors,
  });
}

class ScoreCalculator {
  static QualityResult calculate({
    required int unripeCount,
    int? ripeCount,
    int? freshCount,
    required int overripeCount,
    required int spoiledCount,
  }) {
    final int ripe = ripeCount ?? freshCount ?? 0;
    int total = unripeCount + ripe + overripeCount + spoiledCount;

    if (total == 0) {
      return QualityResult(
        scorePercentage: 0.0,
        gradeLetter: 'N/A',
        recommendationBadge: 'NO TARGET DETECTED',
        aiInsight: 'Align camera viewfinder with tomatoes to analyze harvest quality.',
        shelfLifeDays: 'Unknown',
        optimalTemp: '12°C - 15°C',
        optimalHumidity: '85% - 90%',
        riskAssessment: 'Ensure adequate ambient lighting and steady focus.',
        badgeColor: AppColors.textMuted,
        gradientColors: [AppColors.textMuted, AppColors.borderMuted],
      );
    }

    // Weighted scoring: Ripe = 100%, Unripe = 85%, Overripe = 45%, Spoiled = 0%
    double rawScore =
        ((ripe * 1.0) + (unripeCount * 0.85) + (overripeCount * 0.45) + (spoiledCount * 0.0)) /
            total *
            100;
    double scorePercentage = rawScore.clamp(0.0, 100.0);

    // Grade and agronomic shelf-life logic
    if (spoiledCount > 1 || scorePercentage < 50.0) {
      return QualityResult(
        scorePercentage: scorePercentage,
        gradeLetter: 'C',
        recommendationBadge: 'GRADE C • REJECT / CULL',
        aiInsight: 'High spoilage index detected. Isolate spoiled units immediately to prevent fungal transmission.',
        shelfLifeDays: '< 24 Hours',
        optimalTemp: '8°C - 10°C (Immediate chilling)',
        optimalHumidity: '80% RH',
        riskAssessment: 'Rapid ethylene acceleration & bacterial cross-contamination risk.',
        badgeColor: AppColors.tierSpoiled,
        gradientColors: [const Color(0xFFEF4444), const Color(0xFF991B1B)],
      );
    } else if (scorePercentage >= 85.0) {
      return QualityResult(
        scorePercentage: scorePercentage,
        gradeLetter: 'A+',
        recommendationBadge: 'GRADE A+ • PREMIUM EXPORT',
        aiInsight: 'Peak firmness and optimal pigment maturity. Prime condition for premium retail & export.',
        shelfLifeDays: unripeCount > ripe ? '10 - 14 Days' : '7 - 10 Days',
        optimalTemp: '12°C - 15°C',
        optimalHumidity: '90% RH',
        riskAssessment: 'Low biological risk. Excellent post-harvest resilience.',
        badgeColor: AppColors.tierFresh,
        gradientColors: [const Color(0xFF10B981), const Color(0xFF059669)],
      );
    } else if (scorePercentage >= 70.0) {
      return QualityResult(
        scorePercentage: scorePercentage,
        gradeLetter: 'A',
        recommendationBadge: 'GRADE A • FRESH MARKET',
        aiInsight: 'High market grade. Suitable for regular retail distribution and fresh consumption.',
        shelfLifeDays: '4 - 7 Days',
        optimalTemp: '12°C - 14°C',
        optimalHumidity: '85% - 90% RH',
        riskAssessment: 'Moderate ethylene sensitivity. Monitor ripening progression.',
        badgeColor: AppColors.tierFresh,
        gradientColors: [const Color(0xFF10B981), const Color(0xFF0D9488)],
      );
    } else {
      return QualityResult(
        scorePercentage: scorePercentage,
        gradeLetter: 'B',
        recommendationBadge: 'GRADE B • FOOD PROCESSING',
        aiInsight: 'Advanced ripeness detected. Recommended for immediate culinary preparation, canning, or sauce production.',
        shelfLifeDays: '2 - 3 Days',
        optimalTemp: '10°C - 12°C',
        optimalHumidity: '85% RH',
        riskAssessment: 'Skin elasticity thinning. Process within 48 hours.',
        badgeColor: AppColors.tierOverripe,
        gradientColors: [const Color(0xFFF59E0B), const Color(0xFFD97706)],
      );
    }
  }
}