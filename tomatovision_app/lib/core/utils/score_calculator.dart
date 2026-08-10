import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

class QualityResult {
  final double scorePercentage;
  final String recommendationBadge;
  final String aiInsight;
  final Color badgeColor;

  QualityResult({
    required this.scorePercentage,
    required this.recommendationBadge,
    required this.aiInsight,
    required this.badgeColor,
  });
}

class ScoreCalculator {
  static QualityResult calculate({
    required int unripeCount,
    required int freshCount,
    required int overripeCount,
    required int spoiledCount,
  }) {
    int total = unripeCount + freshCount + overripeCount + spoiledCount;

    if (total == 0) {
      return QualityResult(
        scorePercentage: 0.0,
        recommendationBadge: 'NO DETECTION',
        aiInsight: 'Point camera at tomatoes within the frame to analyze.',
        badgeColor: AppColors.textSecondary,
      );
    }

    double rawScore =
        ((freshCount * 1.0) + (unripeCount * 0.5) + (overripeCount * 0.3)) /
            total *
            100;
    double scorePercentage = rawScore.clamp(0.0, 100.0);

    if (spoiledCount > 2 || scorePercentage < 50) {
      return QualityResult(
        scorePercentage: scorePercentage,
        recommendationBadge: 'GRADE C - REJECT',
        aiInsight: 'High defect ratio detected. Quality below acceptable standard.',
        badgeColor: AppColors.tierSpoiled,
      );
    } else if (scorePercentage >= 75) {
      return QualityResult(
        scorePercentage: scorePercentage,
        recommendationBadge: 'GRADE A - PREMIUM',
        aiInsight: 'Excellent fresh quality for market.',
        badgeColor: AppColors.tierFresh,
      );
    } else {
      return QualityResult(
        scorePercentage: scorePercentage,
        recommendationBadge: 'GRADE B - PROCESS',
        aiInsight: 'Suitable for quick processing or local sale.',
        badgeColor: AppColors.tierOverripe,
      );
    }
  }
}