import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? controller;
  bool isTorchOn = false;

  // Model එකෙන් ලැබෙන Detections (Raw Data)
  List<dynamic> rawDetections = [];

  @override
  void initState() {
    super.initState();
    // ඔබේ Camera Controller Initialization Code එක මෙතැනට...
  }

  // 1. FLASH LIGHT TOGGLE LOGIC
  Future<void> toggleTorch() async {
    if (controller != null && controller!.value.isInitialized) {
      setState(() {
        isTorchOn = !isTorchOn;
      });
      await controller!.setFlashMode(
        isTorchOn ? FlashMode.torch : FlashMode.off,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // 2. NUMBER WARADI PROBLEM FIX: CONFIDENCE SCORE FILTERING
    // 0.65 (65%) ට වඩා වැඩි විශ්වාසනීයත්වයක් ඇති Detections පමණක් Count කරනු ලබයි.
    final validDetections = rawDetections
        .where((d) => (d['confidence'] ?? 0.0) >= 0.65)
        .toList();

    int totalDetected = validDetections.length;
    int fresh = validDetections.where((d) => d['label'] == 'Fresh').length;
    int overripe = validDetections.where((d) => d['label'] == 'Overripe').length;
    int spoiled = validDetections.where((d) => d['label'] == 'Spoiled').length;
    int unripe = validDetections.where((d) => d['label'] == 'Unripe').length;

    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      body: SafeArea(
        child: Column(
          children: [
            // 3. UX IMPROVED CAMERA PREVIEW WITH OVERLAY CONTROLS
            Expanded(
              child: Container(
                margin: const EdgeInsets.all(12),
                child: Stack(
                  children: [
                    // Camera View Container
                    ClipRRect(
                      borderRadius: BorderRadius.circular(24),
                      child: controller != null && controller!.value.isInitialized
                          ? CameraPreview(controller!)
                          : Container(
                              color: Colors.black26,
                              child: const Center(
                                child: CircularProgressIndicator(color: Color(0xFFE55B48)),
                              ),
                            ),
                    ),

                    // Top Overlay: Status Pill + Flash Switch Button
                    Positioned(
                      top: 16,
                      left: 16,
                      right: 16,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          // Lighting Status Badge
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.6),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: const [
                                Icon(Icons.circle, color: Colors.greenAccent, size: 8),
                                SizedBox(width: 8),
                                Text(
                                  "Hold steady — Lighting Good",
                                  style: TextStyle(color: Colors.white, fontSize: 12),
                                ),
                              ],
                            ),
                          ),

                          // Flash Light Switch Button
                          IconButton(
                            style: IconButton.styleFrom(
                              backgroundColor: Colors.black.withOpacity(0.6),
                              shape: const CircleBorder(),
                            ),
                            icon: Icon(
                              isTorchOn ? Icons.flash_on : Icons.flash_off,
                              color: isTorchOn ? Colors.amber : Colors.white,
                            ),
                            onPressed: toggleTorch,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // 4. BOTTOM ANALYTICS PANEL
            Container(
              padding: const EdgeInsets.all(20),
              decoration: const BoxDecoration(
                color: Color(0xFF1E1E1E),
                borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Total Count Title
                  Text(
                    totalDetected > 0
                        ? "Detected: $totalDetected Tomatoes Total"
                        : "No Tomatoes Detected",
                    style: TextStyle(
                      color: totalDetected > 0 ? Colors.white : Colors.white54,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Category Breakdown Grid
                  GridView.count(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    crossAxisCount: 2,
                    childAspectRatio: 3.2,
                    crossAxisSpacing: 10,
                    mainAxisSpacing: 10,
                    children: [
                      _buildStatCard("Unripe", unripe, Colors.blue),
                      _buildStatCard("Fresh", fresh, Colors.green),
                      _buildStatCard("Overripe", overripe, Colors.amber),
                      _buildStatCard("Spoiled", spoiled, Colors.red),
                    ],
                  ),
                  const SizedBox(height: 18),

                  // Bottom Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () {
                            // Rescan Action Logic
                          },
                          icon: const Icon(Icons.refresh, color: Colors.white, size: 20),
                          label: const Text("Rescan", style: TextStyle(color: Colors.white)),
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            side: const BorderSide(color: Colors.white24),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () {
                            // PDF Export Action Logic
                          },
                          icon: const Icon(Icons.picture_as_pdf, color: Colors.white, size: 20),
                          label: const Text("Export PDF", style: TextStyle(color: Colors.white)),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFE55B48),
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                          ),
                        ),
                      ),
                    ],
                  )
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Stat Card Widget Helper
  Widget _buildStatCard(String label, int count, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black26,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Icon(Icons.circle, size: 10, color: color),
              const SizedBox(width: 8),
              Text(label, style: const TextStyle(color: Colors.white70, fontSize: 13)),
            ],
          ),
          Text(
            "$count",
            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15),
          ),
        ],
      ),
    );
  }
}