import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

import '../../core/constants/app_colors.dart';
import '../../core/utils/score_calculator.dart';
import '../../services/api_service.dart';
import '../../services/history_service.dart';
import '../../services/pdf_service.dart';
import '../../services/tflite_helper.dart';
import '../../services/tflite_stub.dart';
import '../widgets/scan_history_sheet.dart';
import '../widgets/stat_pill_card.dart';
import '../widgets/viewfinder_hud_painter.dart';

class ViewfinderScreen extends StatefulWidget {
  const ViewfinderScreen({super.key});

  @override
  State<ViewfinderScreen> createState() => _ViewfinderScreenState();
}

class _ViewfinderScreenState extends State<ViewfinderScreen> with TickerProviderStateMixin {
  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = [];
  int _selectedCameraIndex = 0;
  final TfliteService _tfliteService = getTfliteHelper();

  bool _isCameraInitialized = false;
  bool _isAuditorMode = false;
  bool _isExportingPdf = false;
  bool _isTorchOn = false;
  bool _isScanningAnimation = false;
  bool _isAiBackendOnline = false;
  bool _isAnalyzingPhoto = false;

  Uint8List? _scannedImageBytes;
  Size? _scannedImageSize;
  List<AiDetectionItem> _detections = [];

  late AnimationController _laserController;
  late AnimationController _pulseController;

  // Real detection counts (Starts at 0 until an actual scan is performed)
  int _unripe = 0;
  int _ripe = 0;
  int _overripe = 0;
  int _spoiled = 0;
  bool _hasScanned = false;

  @override
  void initState() {
    super.initState();

    _laserController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2200),
    )..repeat(reverse: true);

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);

    _initEngineAndCamera();
    _checkAiBackendHealth();
  }

  Future<void> _checkAiBackendHealth() async {
    final online = await ApiService.checkServerHealth();
    if (mounted) {
      setState(() => _isAiBackendOnline = online);
    }
  }

  Future<void> _initEngineAndCamera() async {
    await _tfliteService.loadModel();
    await _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _availableCameras = await availableCameras();
      if (_availableCameras.isNotEmpty) {
        await _setupCameraController(_availableCameras[_selectedCameraIndex]);
      }
    } catch (e) {
      debugPrint('Camera stream error: $e');
    }
  }

  Future<void> _setupCameraController(CameraDescription cameraDescription) async {
    if (_cameraController != null) {
      await _cameraController!.dispose();
    }

    _cameraController = CameraController(
      cameraDescription,
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      await _cameraController!.initialize();
      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
          _isTorchOn = false;
        });
      }
    } catch (e) {
      debugPrint('Camera initialize error: $e');
    }
  }

  Future<void> _toggleTorch() async {
    if (_cameraController != null && _cameraController!.value.isInitialized) {
      try {
        final newTorchState = !_isTorchOn;
        await _cameraController!.setFlashMode(
          newTorchState ? FlashMode.torch : FlashMode.off,
        );
        setState(() => _isTorchOn = newTorchState);
      } catch (e) {
        debugPrint('Torch toggle error: $e');
      }
    }
  }

  Future<void> _switchCamera() async {
    if (_availableCameras.length > 1) {
      _selectedCameraIndex = (_selectedCameraIndex + 1) % _availableCameras.length;
      setState(() => _isCameraInitialized = false);
      await _setupCameraController(_availableCameras[_selectedCameraIndex]);
    }
  }

  // Real Image Selection & YOLOv8 AI Analysis
  Future<void> _pickAndAnalyzePhoto({ImageSource source = ImageSource.gallery}) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile == null) return;

    setState(() {
      _isAnalyzingPhoto = true;
      _isScanningAnimation = true;
    });

    try {
      final imageBytes = await pickedFile.readAsBytes();
      
      // Decode image dimensions for HUD coordinates
      final decodedImage = await decodeImageFromList(imageBytes);
      final imgSize = Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());

      final response = await ApiService.analyzeImageBytes(imageBytes);

      if (response != null && response.success) {
        if (mounted) {
          setState(() {
            _scannedImageBytes = imageBytes;
            _scannedImageSize = imgSize;
            _detections = response.detections;
            _ripe = response.ripe;
            _unripe = response.unripe;
            _overripe = response.overripe;
            _spoiled = response.spoiled;
            _hasScanned = true;
            _isAiBackendOnline = true;
          });

          final result = ScoreCalculator.calculate(
            unripeCount: _unripe,
            ripeCount: _ripe,
            overripeCount: _overripe,
            spoiledCount: _spoiled,
          );

          HistoryService.addRecord(
            ScanRecord(
              id: 'YOLO-${DateTime.now().millisecondsSinceEpoch.toString().substring(7)}',
              timestamp: DateTime.now(),
              result: result,
              unripe: _unripe,
              fresh: _ripe,
              overripe: _overripe,
              spoiled: _spoiled,
              isAuditorMode: _isAuditorMode,
            ),
          );

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                '🎯 AI Scan: ${response.totalDetected} Tomatoes Detected (${response.ripe} Ripe, ${response.unripe} Unripe, ${response.overripe} Overripe, ${response.spoiled} Spoiled)',
                style: GoogleFonts.inter(fontWeight: FontWeight.w600),
              ),
              backgroundColor: AppColors.tierFresh,
              behavior: SnackBarBehavior.floating,
              duration: const Duration(seconds: 4),
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Photo scan error: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isAnalyzingPhoto = false;
          _isScanningAnimation = false;
        });
      }
    }
  }

  // Live Camera Capture & Instant AI Analysis
  Future<void> _captureAndAnalyzeLiveCamera() async {
    if (_isAnalyzingPhoto) return;

    setState(() {
      _isAnalyzingPhoto = true;
      _isScanningAnimation = true;
    });

    try {
      Uint8List? imageBytes;
      Size? imgSize;

      if (_cameraController != null && _cameraController!.value.isInitialized) {
        final xFile = await _cameraController!.takePicture();
        imageBytes = await xFile.readAsBytes();
        final decodedImage = await decodeImageFromList(imageBytes);
        imgSize = Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
      } else {
        final picker = ImagePicker();
        final pickedFile = await picker.pickImage(source: ImageSource.camera);
        if (pickedFile != null) {
          imageBytes = await pickedFile.readAsBytes();
          final decodedImage = await decodeImageFromList(imageBytes);
          imgSize = Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
        }
      }

      if (imageBytes != null && imgSize != null) {
        final response = await ApiService.analyzeImageBytes(imageBytes);
        if (response != null && response.success && mounted) {
          setState(() {
            _scannedImageBytes = imageBytes;
            _scannedImageSize = imgSize;
            _detections = response.detections;
            _ripe = response.ripe;
            _unripe = response.unripe;
            _overripe = response.overripe;
            _spoiled = response.spoiled;
            _hasScanned = true;
            _isAiBackendOnline = true;
          });

          final result = ScoreCalculator.calculate(
            unripeCount: _unripe,
            ripeCount: _ripe,
            overripeCount: _overripe,
            spoiledCount: _spoiled,
          );

          HistoryService.addRecord(
            ScanRecord(
              id: 'SCAN-${DateTime.now().millisecondsSinceEpoch.toString().substring(7)}',
              timestamp: DateTime.now(),
              result: result,
              unripe: _unripe,
              fresh: _ripe,
              overripe: _overripe,
              spoiled: _spoiled,
              isAuditorMode: _isAuditorMode,
            ),
          );

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                '🎯 Live Scan: ${response.totalDetected} Tomatoes Detected (${response.ripe} Ripe, ${response.unripe} Unripe, ${response.overripe} Overripe, ${response.spoiled} Spoiled)',
                style: GoogleFonts.inter(fontWeight: FontWeight.w600),
              ),
              backgroundColor: AppColors.tierFresh,
              behavior: SnackBarBehavior.floating,
              duration: const Duration(seconds: 4),
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Live camera capture error: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isAnalyzingPhoto = false;
          _isScanningAnimation = false;
        });
      }
    }
  }

  // Trigger Re-scan with real YOLO backend
  Future<void> _triggerRealRescan() async {
    if (_scannedImageBytes != null) {
      setState(() {
        _isAnalyzingPhoto = true;
        _isScanningAnimation = true;
      });

      final response = await ApiService.analyzeImageBytes(_scannedImageBytes!);
      if (response != null && response.success && mounted) {
        setState(() {
          _ripe = response.ripe;
          _unripe = response.unripe;
          _overripe = response.overripe;
          _spoiled = response.spoiled;
          _detections = response.detections;
          _hasScanned = true;
          _isAiBackendOnline = true;
        });
      }

      if (mounted) {
        setState(() {
          _isAnalyzingPhoto = false;
          _isScanningAnimation = false;
        });
      }
    } else {
      _captureAndAnalyzeLiveCamera();
    }
  }

  void _showHistoryModal() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => ScanHistorySheet(
        onSelectRecord: (record) {
          setState(() {
            _unripe = record.unripe;
            _ripe = record.fresh;
            _overripe = record.overripe;
            _spoiled = record.spoiled;
            _hasScanned = true;
            _isAuditorMode = record.isAuditorMode;
          });
        },
      ),
    );
  }

  Future<void> _handlePdfExport(QualityResult result) async {
    if (!_hasScanned) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please scan or upload a tomato photo before exporting report.'),
          backgroundColor: AppColors.cardBackground,
          behavior: SnackBarBehavior.floating,
        ),
      );
      return;
    }

    setState(() => _isExportingPdf = true);
    try {
      await PdfService.generateAndExportReport(
        result: result,
        unripe: _unripe,
        fresh: _ripe,
        overripe: _overripe,
        spoiled: _spoiled,
        isAuditorMode: _isAuditorMode,
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('PDF Export Notice: $e'),
            backgroundColor: AppColors.cardBackground,
          ),
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
    _laserController.dispose();
    _pulseController.dispose();
    _cameraController?.dispose();
    _tfliteService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final result = ScoreCalculator.calculate(
      unripeCount: _unripe,
      ripeCount: _ripe,
      overripeCount: _overripe,
      spoiledCount: _spoiled,
    );

    final int total = _unripe + _ripe + _overripe + _spoiled;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildTopNavBar(),
            Expanded(
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      // 1. Scanned Image Layer OR Camera Preview OR Upload Placeholder
                      GestureDetector(
                        onTap: _isAnalyzingPhoto ? null : _captureAndAnalyzeLiveCamera,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            if (_scannedImageBytes != null)
                              Image.memory(
                                _scannedImageBytes!,
                                fit: BoxFit.contain,
                                alignment: Alignment.center,
                              )
                            else if (_isCameraInitialized && _cameraController != null)
                              FittedBox(
                                fit: BoxFit.cover,
                                child: SizedBox(
                                  width: _cameraController!.value.previewSize?.height ?? 1,
                                  height: _cameraController!.value.previewSize?.width ?? 1,
                                  child: CameraPreview(_cameraController!),
                                ),
                              )
                            else
                              _buildUploadPromptPlaceholder(),
                          ],
                        ),
                      ),

                      // 2. Real HUD Overlay (Laser Scan & Real YOLO Detections)
                      AnimatedBuilder(
                        animation: Listenable.merge([_laserController, _pulseController]),
                        builder: (context, _) {
                          return CustomPaint(
                            painter: ViewfinderHudPainter(
                              scanProgress: _laserController.value,
                              pulseValue: _pulseController.value,
                              isAuditorMode: _isAuditorMode,
                              detections: _detections,
                              originalImageSize: _scannedImageSize,
                            ),
                          );
                        },
                      ),

                      // 3. Center Live Camera Shutter Button (Visible when on live camera)
                      if (_scannedImageBytes == null)
                        Positioned(
                          bottom: 20,
                          left: 0,
                          right: 0,
                          child: Center(
                            child: GestureDetector(
                              onTap: _isAnalyzingPhoto ? null : _captureAndAnalyzeLiveCamera,
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
                                decoration: BoxDecoration(
                                  gradient: const LinearGradient(
                                    colors: [AppColors.brandPrimary, AppColors.brandSecondary],
                                  ),
                                  borderRadius: BorderRadius.circular(30),
                                  boxShadow: const [
                                    BoxShadow(color: AppColors.brandGlow, blurRadius: 16, offset: Offset(0, 4)),
                                  ],
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    _isAnalyzingPhoto
                                        ? const SizedBox(
                                            width: 18,
                                            height: 18,
                                            child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                                          )
                                        : const Icon(Icons.camera_alt_rounded, color: Colors.white, size: 20),
                                    const SizedBox(width: 8),
                                    Text(
                                      _isAnalyzingPhoto ? 'SCANNING...' : 'TAP TO SCAN CAMERA',
                                      style: GoogleFonts.plusJakartaSans(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 13,
                                        letterSpacing: 0.5,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ),

                      // 3. Top HUD Pill Controls (Backend Status, Upload, Flash, Flip)
                      Positioned(
                        top: 14,
                        left: 14,
                        right: 14,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            // AI Backend & Scanning Status Badge
                            GestureDetector(
                              onTap: _checkAiBackendHealth,
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: 0.75),
                                  borderRadius: BorderRadius.circular(20),
                                  border: Border.all(color: Colors.white12),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      width: 8,
                                      height: 8,
                                      decoration: BoxDecoration(
                                        color: AppColors.tierFresh,
                                        shape: BoxShape.circle,
                                        boxShadow: [
                                          BoxShadow(
                                            color: AppColors.tierFresh.withValues(alpha: 0.6),
                                            blurRadius: 6,
                                          ),
                                        ],
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Text(
                                      _isAnalyzingPhoto
                                          ? 'Scanning with AI...'
                                          : 'AI Engine: Online (Ready)',
                                      style: GoogleFonts.inter(
                                        color: Colors.white,
                                        fontSize: 11,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),

                            // Right Action Icons (Upload Image + Flash + Flip)
                            Row(
                              children: [
                                _buildGlassCircleButton(
                                  icon: Icons.add_photo_alternate_rounded,
                                  color: Colors.white,
                                  tooltip: 'Upload Tomato Photo (Real YOLOv8 AI Scan)',
                                  onTap: () => _pickAndAnalyzePhoto(source: ImageSource.gallery),
                                ),
                                if (_isCameraInitialized) ...[
                                  const SizedBox(width: 8),
                                  _buildGlassCircleButton(
                                    icon: _isTorchOn ? Icons.flash_on_rounded : Icons.flash_off_rounded,
                                    color: _isTorchOn ? Colors.amber : Colors.white70,
                                    onTap: _toggleTorch,
                                  ),
                                  if (_availableCameras.length > 1) ...[
                                    const SizedBox(width: 8),
                                    _buildGlassCircleButton(
                                      icon: Icons.flip_camera_ios_rounded,
                                      color: Colors.white70,
                                      onTap: _switchCamera,
                                    ),
                                  ],
                                ],
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            _buildInteractiveDashboard(result, total),
          ],
        ),
      ),
    );
  }

  Widget _buildTopNavBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [AppColors.brandPrimary, AppColors.brandSecondary],
                  ),
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: const [
                    BoxShadow(color: AppColors.brandGlow, blurRadius: 10, offset: Offset(0, 3)),
                  ],
                ),
                child: const Icon(Icons.remove_red_eye_rounded, color: Colors.white, size: 18),
              ),
              const SizedBox(width: 10),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'TomatoVision AI',
                    style: GoogleFonts.plusJakartaSans(
                      color: AppColors.textPrimary,
                      fontSize: 17,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  Text(
                    _isAuditorMode ? 'Auditor Inspection Suite' : 'Agro-Vision Quality Suite',
                    style: GoogleFonts.inter(
                      color: _isAuditorMode ? AppColors.auditorBlue : AppColors.textMuted,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ],
          ),

          // Mode Toggle
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(22),
              border: Border.all(color: AppColors.borderMuted),
            ),
            child: Row(
              children: [
                _buildModePill('Consumer', !_isAuditorMode, () {
                  setState(() => _isAuditorMode = false);
                }),
                _buildModePill('Auditor', _isAuditorMode, () {
                  setState(() => _isAuditorMode = true);
                }),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModePill(String label, bool isSelected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOutCubic,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected
              ? (_isAuditorMode ? AppColors.auditorBlue : AppColors.brandPrimary)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            if (isSelected)
              BoxShadow(
                color: (_isAuditorMode ? AppColors.auditorGlow : AppColors.brandGlow),
                blurRadius: 8,
              ),
          ],
        ),
        child: Text(
          label,
          style: GoogleFonts.inter(
            color: isSelected ? Colors.white : AppColors.textSecondary,
            fontSize: 11,
            fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
          ),
        ),
      ),
    );
  }

  Widget _buildGlassCircleButton({
    required IconData icon,
    required Color color,
    required VoidCallback onTap,
    String? tooltip,
  }) {
    return Tooltip(
      message: tooltip ?? '',
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.black.withValues(alpha: 0.68),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white12),
          ),
          child: Icon(icon, size: 18, color: color),
        ),
      ),
    );
  }

  Widget _buildUploadPromptPlaceholder() {
    return Container(
      color: const Color(0xFF0F1524),
      padding: const EdgeInsets.all(24),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: AppColors.surface,
                shape: BoxShape.circle,
                border: Border.all(color: AppColors.borderMuted),
                boxShadow: const [
                  BoxShadow(color: AppColors.brandGlow, blurRadius: 16),
                ],
              ),
              child: const Icon(
                Icons.add_a_photo_rounded,
                size: 42,
                color: AppColors.brandPrimary,
              ),
            ),
            const SizedBox(height: 18),
            Text(
              'SELECT OR CAPTURE TOMATO PHOTO',
              textAlign: TextAlign.center,
              style: GoogleFonts.plusJakartaSans(
                color: AppColors.textPrimary,
                fontSize: 14,
                letterSpacing: 1.1,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Upload any tomato image to run real-time YOLOv8 AI ripeness & quality detection',
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(
                color: AppColors.textMuted,
                fontSize: 11.5,
              ),
            ),
            const SizedBox(height: 18),
            ElevatedButton.icon(
              onPressed: () => _pickAndAnalyzePhoto(source: ImageSource.gallery),
              icon: const Icon(Icons.file_upload_outlined, size: 18, color: Colors.white),
              label: Text(
                'Upload Tomato Image',
                style: GoogleFonts.plusJakartaSans(
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                  color: Colors.white,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.brandPrimary,
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInteractiveDashboard(QualityResult result, int total) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 14),
      decoration: const BoxDecoration(
        color: AppColors.surfaceGlass,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        border: Border(
          top: BorderSide(color: AppColors.borderMuted, width: 1.2),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Row 1: Radial Gauge + Grade Banner & Insight
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              CircularPercentIndicator(
                radius: 38.0,
                lineWidth: 7.0,
                percent: _hasScanned ? (result.scorePercentage / 100.0).clamp(0.0, 1.0) : 0.0,
                center: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      _hasScanned ? '${result.scorePercentage.round()}%' : '--%',
                      style: GoogleFonts.plusJakartaSans(
                        color: AppColors.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    Text(
                      'Quality',
                      style: GoogleFonts.inter(
                        color: AppColors.textMuted,
                        fontSize: 9,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
                progressColor: _hasScanned ? result.badgeColor : AppColors.borderMuted,
                backgroundColor: AppColors.cardElevated,
                circularStrokeCap: CircularStrokeCap.round,
                animation: true,
                animationDuration: 500,
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                      decoration: BoxDecoration(
                        color: _hasScanned ? result.badgeColor.withValues(alpha: 0.12) : AppColors.cardElevated,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: _hasScanned ? result.badgeColor.withValues(alpha: 0.4) : AppColors.borderMuted,
                          width: 1.2,
                        ),
                      ),
                      child: Text(
                        _hasScanned ? result.recommendationBadge : 'READY TO SCAN',
                        style: GoogleFonts.plusJakartaSans(
                          color: _hasScanned ? result.badgeColor : AppColors.textSecondary,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.4,
                        ),
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _hasScanned
                          ? result.aiInsight
                          : 'Select or upload a tomato photo to perform real-time YOLOv8 AI inspection.',
                      style: GoogleFonts.inter(
                        color: AppColors.textSecondary,
                        fontSize: 11.5,
                        height: 1.25,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
            ],
          ),

          // Row 2: Shelf Life & Storage Banner
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: AppColors.cardBackground,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.borderMuted),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.access_time_rounded, size: 15, color: AppColors.tierUnripe),
                    const SizedBox(width: 6),
                    Text(
                      _hasScanned ? 'Est. Shelf-Life: ${result.shelfLifeDays}' : 'Est. Shelf-Life: --',
                      style: GoogleFonts.inter(
                        color: AppColors.textPrimary,
                        fontSize: 11.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
                Row(
                  children: [
                    const Icon(Icons.thermostat_rounded, size: 15, color: AppColors.tierFresh),
                    const SizedBox(width: 4),
                    Text(
                      _hasScanned ? result.optimalTemp : '12°C - 15°C',
                      style: GoogleFonts.inter(
                        color: AppColors.textSecondary,
                        fontSize: 11,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Row 3: Ripeness Breakdown Pill Cards (2x2 Grid)
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: StatPillCard(
                  label: 'Ripe',
                  subtitle: 'Optimal Ripeness',
                  count: _ripe,
                  totalCount: total,
                  color: AppColors.tierFresh,
                  icon: Icons.check_circle_outline_rounded,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: StatPillCard(
                  label: 'Unripe',
                  subtitle: 'Breaker / Green',
                  count: _unripe,
                  totalCount: total,
                  color: AppColors.tierUnripe,
                  icon: Icons.eco_outlined,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: StatPillCard(
                  label: 'Overripe',
                  subtitle: 'Processing Ready',
                  count: _overripe,
                  totalCount: total,
                  color: AppColors.tierOverripe,
                  icon: Icons.wb_sunny_outlined,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: StatPillCard(
                  label: 'Spoiled',
                  subtitle: 'Cull Required',
                  count: _spoiled,
                  totalCount: total,
                  color: AppColors.tierSpoiled,
                  icon: Icons.warning_amber_rounded,
                ),
              ),
            ],
          ),


          // Row 4: Action Deck (History + Upload Photo + Rescan + Export PDF)
          const SizedBox(height: 14),
          Row(
            children: [
              // History Modal Trigger
              Container(
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.borderMuted),
                ),
                child: IconButton(
                  onPressed: _showHistoryModal,
                  tooltip: 'Scan History',
                  icon: const Icon(Icons.history_rounded, color: AppColors.textPrimary, size: 20),
                ),
              ),
              const SizedBox(width: 8),

              // Upload Photo Button
              Container(
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.tierFresh.withValues(alpha: 0.5)),
                ),
                child: IconButton(
                  onPressed: _isAnalyzingPhoto
                      ? null
                      : () => _pickAndAnalyzePhoto(source: ImageSource.gallery),
                  tooltip: 'Upload Tomato Photo (Live YOLOv8 AI)',
                  icon: _isAnalyzingPhoto
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(color: AppColors.tierFresh, strokeWidth: 2),
                        )
                      : const Icon(Icons.photo_library_rounded, color: AppColors.tierFresh, size: 20),
                ),
              ),
              const SizedBox(width: 8),

              // Scan / Rescan Action Button
              Expanded(
                flex: 42,
                child: ElevatedButton.icon(
                  onPressed: _isAnalyzingPhoto
                      ? null
                      : (_scannedImageBytes == null ? _captureAndAnalyzeLiveCamera : _triggerRealRescan),
                  icon: _isAnalyzingPhoto
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                        )
                      : Icon(
                          _scannedImageBytes == null ? Icons.camera_alt_rounded : Icons.refresh_rounded,
                          size: 18,
                          color: Colors.white,
                        ),
                  label: Text(
                    _scannedImageBytes == null ? 'Scan Live' : 'Rescan',
                    style: GoogleFonts.plusJakartaSans(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                      fontSize: 13,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.brandPrimary,
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    elevation: 0,
                  ),
                ),
              ),
              const SizedBox(width: 8),

              // Export PDF Button
              Expanded(
                flex: 52,
                child: ElevatedButton.icon(
                  onPressed: _isExportingPdf ? null : () => _handlePdfExport(result),
                  icon: _isExportingPdf
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                        )
                      : const Icon(Icons.picture_as_pdf_rounded, size: 18, color: Colors.white),
                  label: Text(
                    _isExportingPdf
                        ? 'Exporting...'
                        : (_isAuditorMode ? 'Audit Certificate' : 'Export Report'),
                    style: GoogleFonts.plusJakartaSans(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 13,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isAuditorMode ? AppColors.auditorBlue : AppColors.brandPrimary,
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
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
}

