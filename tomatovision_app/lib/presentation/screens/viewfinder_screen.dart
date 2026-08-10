import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

import '../../core/constants/app_colors.dart';
import '../../core/utils/score_calculator.dart';
import '../../services/pdf_service.dart';
import '../../services/tflite_helper.dart';
import '../../services/tflite_stub.dart';

class ViewfinderScreen extends StatefulWidget {
  const ViewfinderScreen({super.key});

  @override
  State<ViewfinderScreen> createState() => _ViewfinderScreenState();
}

class _ViewfinderScreenState extends State<ViewfinderScreen> {
  CameraController? _cameraController;
  final TfliteService _tfliteService = getTfliteHelper();
  bool _isCameraInitialized = false;
  bool _isAuditorMode = false;
  bool _isExportingPdf = false;

  int _unripe = 0;
  int _fresh = 7;
  int _overripe = 3;
  int _spoiled = 1;

  @override
  void initState() {
    super.initState();
    _initEngineAndCamera();
  }

  Future<void> _initEngineAndCamera() async {
    await _tfliteService.loadModel();
    await _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        _cameraController = CameraController(
          cameras.first,
          ResolutionPreset.high,
          enableAudio: false,
        );
        await _cameraController!.initialize();
        if (mounted) {
          setState(() {
            _isCameraInitialized = true;
          });
        }
      }
    } catch (e) {
      debugPrint('Camera permission/stream error: $e');
    }
  }

  void _randomizeMockScan() {
    final rand = Random();
    setState(() {
      _unripe = rand.nextInt(3);
      _fresh = rand.nextInt(6) + 4;
      _overripe = rand.nextInt(4);
      _spoiled = rand.nextInt(2);
    });
  }

  Future<void> _handlePdfExport(QualityResult result) async {
    setState(() => _isExportingPdf = true);
    try {
      await PdfService.generateAndExportReport(
        result: result,
        unripe: _unripe,
        fresh: _fresh,
        overripe: _overripe,
        spoiled: _spoiled,
        isAuditorMode: _isAuditorMode,
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to export PDF: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isExportingPdf = false);
      }
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _tfliteService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final result = ScoreCalculator.calculate(
      unripeCount: _unripe,
      freshCount: _fresh,
      overripeCount: _overripe,
      spoiledCount: _spoiled,
    );

    final int total = _unripe + _fresh + _overripe + _spoiled;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeaderBar(),
            Expanded(
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      if (_isCameraInitialized && _cameraController != null)
                        FittedBox(
                          fit: BoxFit.cover,
                          child: SizedBox(
                            width: _cameraController!.value.previewSize?.height ?? 1,
                            height: _cameraController!.value.previewSize?.width ?? 1,
                            child: CameraPreview(_cameraController!),
                          ),
                        )
                      else
                        _buildCameraPlaceholder(),

                      Positioned(
                        top: 16,
                        left: 0,
                        right: 0,
                        child: Center(
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                              vertical: 8,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.65),
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(color: Colors.white24),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Container(
                                  width: 8,
                                  height: 8,
                                  decoration: const BoxDecoration(
                                    color: AppColors.tierFresh,
                                    shape: BoxShape.circle,
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Text(
                                  _tfliteService.isModelLoaded
                                      ? 'Hold steady — Lighting Good'
                                      : 'Initializing AI Model...',
                                  style: GoogleFonts.inter(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            _buildResultsBottomSheet(result, total),
          ],
        ),
      ),
    );
  }

  Widget _buildHeaderBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'TomatoVision AI',
                style: GoogleFonts.inter(
                  color: AppColors.textPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Row(
                children: [
                  Container(
                    width: 6,
                    height: 6,
                    decoration: const BoxDecoration(
                      color: AppColors.tierFresh,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    _isAuditorMode ? 'Auditor Inspection Active' : 'Offline AI Active',
                    style: GoogleFonts.inter(
                      color: AppColors.textSecondary,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ],
          ),
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: AppColors.cardBackground,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: AppColors.borderMuted),
            ),
            child: Row(
              children: [
                _buildModeTab('Consumer', !_isAuditorMode, () {
                  setState(() => _isAuditorMode = false);
                }),
                _buildModeTab('Auditor', _isAuditorMode, () {
                  setState(() => _isAuditorMode = true);
                }),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModeTab(String label, bool isSelected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? AppColors.tierSpoiled : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          label,
          style: GoogleFonts.inter(
            color: isSelected ? Colors.white : AppColors.textSecondary,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }

  Widget _buildCameraPlaceholder() {
    return Container(
      color: const Color(0xFF14161B),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.camera_alt_outlined,
              size: 54,
              color: AppColors.textSecondary.withOpacity(0.4),
            ),
            const SizedBox(height: 16),
            Text(
              'POINT CAMERA AT TOMATOES',
              style: GoogleFonts.inter(
                color: AppColors.textSecondary.withOpacity(0.7),
                fontSize: 12,
                letterSpacing: 1.2,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsBottomSheet(QualityResult result, int totalTomatoes) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: const BoxDecoration(
        color: AppColors.cardBackground,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        border: Border(
          top: BorderSide(color: AppColors.borderMuted, width: 1),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              CircularPercentIndicator(
                radius: 40.0,
                lineWidth: 8.0,
                percent: (result.scorePercentage / 100.0).clamp(0.0, 1.0),
                center: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      '${result.scorePercentage.round()}%',
                      style: GoogleFonts.inter(
                        color: AppColors.textPrimary,
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    Text(
                      'Quality',
                      style: GoogleFonts.inter(
                        color: AppColors.textSecondary,
                        fontSize: 9,
                      ),
                    ),
                  ],
                ),
                progressColor: result.badgeColor,
                backgroundColor: AppColors.background,
                circularStrokeCap: CircularStrokeCap.round,
                animation: true,
                animationDuration: 400,
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: result.badgeColor.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: result.badgeColor,
                          width: 1.2,
                        ),
                      ),
                      child: Text(
                        result.recommendationBadge,
                        style: GoogleFonts.inter(
                          color: result.badgeColor,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      result.aiInsight,
                      style: GoogleFonts.inter(
                        color: AppColors.textSecondary,
                        fontSize: 12,
                        height: 1.3,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          // Auditor Mode Extra Info Card
          if (_isAuditorMode) ...[
            const SizedBox(height: 14),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.blue.withOpacity(0.3)),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.verified, size: 16, color: Colors.blue),
                      const SizedBox(width: 6),
                      Text(
                        'Audit Batch: #TV-84920',
                        style: GoogleFonts.inter(
                          color: Colors.blue,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  Text(
                    'Confidence: 96.4%',
                    style: GoogleFonts.inter(
                      color: AppColors.textSecondary,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
          ],

          const SizedBox(height: 16),

          Text(
            'Detected: $totalTomatoes Tomatoes Total',
            style: GoogleFonts.inter(
              color: AppColors.textPrimary,
              fontSize: 14,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 12),

          Row(
            children: [
              Expanded(child: _buildPill('🔵 Unripe', _unripe, AppColors.tierUnripe)),
              const SizedBox(width: 10),
              Expanded(child: _buildPill('🟢 Fresh', _fresh, AppColors.tierFresh)),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(child: _buildPill('🟡 Overripe', _overripe, AppColors.tierOverripe)),
              const SizedBox(width: 10),
              Expanded(child: _buildPill('🔴 Spoiled', _spoiled, AppColors.tierSpoiled)),
            ],
          ),

          const SizedBox(height: 20),

          Row(
            children: [
              Expanded(
                flex: 45,
                child: OutlinedButton.icon(
                  onPressed: _randomizeMockScan,
                  icon: const Icon(Icons.sync, size: 18, color: AppColors.textPrimary),
                  label: Text(
                    'Rescan',
                    style: GoogleFonts.inter(
                      color: AppColors.textPrimary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    side: const BorderSide(color: AppColors.borderMuted, width: 1.5),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 55,
                child: ElevatedButton.icon(
                  onPressed: _isExportingPdf ? null : () => _handlePdfExport(result),
                  icon: _isExportingPdf
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                        )
                      : const Icon(Icons.picture_as_pdf_outlined, size: 18, color: Colors.white),
                  label: Text(
                    _isExportingPdf ? 'Exporting...' : (_isAuditorMode ? 'Audit PDF' : 'Export PDF'),
                    style: GoogleFonts.inter(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.tierSpoiled,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: 0,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPill(String label, int count, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: AppColors.background,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.4), width: 1),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: GoogleFonts.inter(
              color: AppColors.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
          Text(
            '$count',
            style: GoogleFonts.inter(
              color: Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}