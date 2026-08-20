import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../../core/constants/app_colors.dart';
import '../../services/history_service.dart';
import '../../services/pdf_service.dart';

class ScanHistorySheet extends StatefulWidget {
  final Function(ScanRecord record) onSelectRecord;

  const ScanHistorySheet({super.key, required this.onSelectRecord});

  @override
  State<ScanHistorySheet> createState() => _ScanHistorySheetState();
}

class _ScanHistorySheetState extends State<ScanHistorySheet> {
  String? _exportingId;

  Future<void> _exportRecordPdf(ScanRecord record) async {
    setState(() => _exportingId = record.id);
    try {
      await PdfService.generateAndExportReport(
        result: record.result,
        unripe: record.unripe,
        fresh: record.fresh,
        overripe: record.overripe,
        spoiled: record.spoiled,
        isAuditorMode: record.isAuditorMode,
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Export error: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _exportingId = null);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final records = HistoryService.records;

    return Container(
      height: MediaQuery.of(context).size.height * 0.72,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        border: Border(top: BorderSide(color: AppColors.borderMuted, width: 1.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Drag Handle
          Center(
            child: Container(
              width: 44,
              height: 4,
              decoration: BoxDecoration(
                color: AppColors.borderMuted,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Header
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: AppColors.brandPrimary.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(Icons.history_rounded, color: AppColors.brandPrimary, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Audit & Scan History',
                        style: GoogleFonts.plusJakartaSans(
                          color: AppColors.textPrimary,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        '${records.length} logged batches in session',
                        style: GoogleFonts.inter(
                          color: AppColors.textMuted,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close_rounded, color: AppColors.textSecondary),
              ),
            ],
          ),
          const SizedBox(height: 16),
          const Divider(color: AppColors.borderMuted, height: 1),
          const SizedBox(height: 12),

          // List or Empty
          Expanded(
            child: records.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.inventory_2_outlined, size: 48, color: AppColors.textMuted.withOpacity(0.5)),
                        const SizedBox(height: 12),
                        Text(
                          'No Batch Scans Yet',
                          style: GoogleFonts.plusJakartaSans(
                            color: AppColors.textSecondary,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Scans taken in the viewfinder will appear here.',
                          style: GoogleFonts.inter(color: AppColors.textMuted, fontSize: 12),
                        ),
                      ],
                    ),
                  )
                : ListView.separated(
                    itemCount: records.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 12),
                    itemBuilder: (context, index) {
                      final item = records[index];
                      final isExporting = _exportingId == item.id;
                      final timeStr = DateFormat('MMM dd, hh:mm a').format(item.timestamp);

                      return Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: AppColors.cardBackground,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: AppColors.borderMuted),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                      decoration: BoxDecoration(
                                        color: item.result.badgeColor.withOpacity(0.15),
                                        borderRadius: BorderRadius.circular(8),
                                        border: Border.all(color: item.result.badgeColor.withOpacity(0.5)),
                                      ),
                                      child: Text(
                                        'Grade ${item.result.gradeLetter}',
                                        style: GoogleFonts.plusJakartaSans(
                                          color: item.result.badgeColor,
                                          fontSize: 11,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Text(
                                      '#${item.id}',
                                      style: GoogleFonts.inter(
                                        color: AppColors.textPrimary,
                                        fontSize: 13,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ],
                                ),
                                Text(
                                  timeStr,
                                  style: GoogleFonts.inter(color: AppColors.textMuted, fontSize: 11),
                                ),
                              ],
                            ),
                            const SizedBox(height: 10),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'Quality Score: ${item.result.scorePercentage.round()}%',
                                  style: GoogleFonts.inter(
                                    color: item.result.badgeColor,
                                    fontSize: 13,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Text(
                                  'Total: ${item.totalCount} Units',
                                  style: GoogleFonts.inter(color: AppColors.textSecondary, fontSize: 12),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            // Mini Chips
                            Wrap(
                              spacing: 6,
                              runSpacing: 4,
                              children: [
                                _buildMiniCountChip('Ripe', item.fresh, AppColors.tierFresh),
                                _buildMiniCountChip('Unripe', item.unripe, AppColors.tierUnripe),
                                _buildMiniCountChip('Overripe', item.overripe, AppColors.tierOverripe),
                                _buildMiniCountChip('Spoiled', item.spoiled, AppColors.tierSpoiled),
                              ],
                            ),
                            const SizedBox(height: 12),
                            // Actions
                            Row(
                              children: [
                                Expanded(
                                  child: OutlinedButton.icon(
                                    onPressed: () {
                                      widget.onSelectRecord(item);
                                      Navigator.pop(context);
                                    },
                                    icon: const Icon(Icons.visibility_outlined, size: 16, color: AppColors.textPrimary),
                                    label: Text(
                                      'Load In View',
                                      style: GoogleFonts.inter(color: AppColors.textPrimary, fontSize: 12, fontWeight: FontWeight.w600),
                                    ),
                                    style: OutlinedButton.styleFrom(
                                      padding: const EdgeInsets.symmetric(vertical: 8),
                                      side: const BorderSide(color: AppColors.borderMuted),
                                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Expanded(
                                  child: ElevatedButton.icon(
                                    onPressed: isExporting ? null : () => _exportRecordPdf(item),
                                    icon: isExporting
                                        ? const SizedBox(
                                            width: 14,
                                            height: 14,
                                            child: CircularProgressIndicator(color: Colors.white, strokeWidth: 1.5),
                                          )
                                        : const Icon(Icons.picture_as_pdf_outlined, size: 16, color: Colors.white),
                                    label: Text(
                                      'Export PDF',
                                      style: GoogleFonts.inter(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                                    ),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: AppColors.brandPrimary,
                                      padding: const EdgeInsets.symmetric(vertical: 8),
                                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildMiniCountChip(String label, int count, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text(
        '$label: $count',
        style: GoogleFonts.inter(color: color, fontSize: 10, fontWeight: FontWeight.w600),
      ),
    );
  }
}

