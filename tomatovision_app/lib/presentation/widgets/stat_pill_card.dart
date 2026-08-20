import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../core/constants/app_colors.dart';

class StatPillCard extends StatefulWidget {
  final String label;
  final String subtitle;
  final int count;
  final int totalCount;
  final Color color;
  final IconData icon;
  final VoidCallback? onTap;

  const StatPillCard({
    super.key,
    required this.label,
    required this.subtitle,
    required this.count,
    required this.totalCount,
    required this.color,
    required this.icon,
    this.onTap,
  });

  @override
  State<StatPillCard> createState() => _StatPillCardState();
}

class _StatPillCardState extends State<StatPillCard> {
  bool _isPressed = false;

  double get _ratio => widget.totalCount > 0 ? (widget.count / widget.totalCount) : 0.0;
  String get _percentageStr => '${(_ratio * 100).toStringAsFixed(0)}%';

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _isPressed = true),
      onTapUp: (_) => setState(() => _isPressed = false),
      onTapCancel: () => setState(() => _isPressed = false),
      onTap: widget.onTap,
      child: AnimatedScale(
        scale: _isPressed ? 0.96 : 1.0,
        duration: const Duration(milliseconds: 100),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: widget.count > 0 ? widget.color.withOpacity(0.35) : AppColors.borderMuted,
              width: 1.2,
            ),
            boxShadow: [
              if (widget.count > 0)
                BoxShadow(
                  color: widget.color.withOpacity(0.08),
                  blurRadius: 12,
                  offset: const Offset(0, 4),
                ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(6),
                        decoration: BoxDecoration(
                          color: widget.color.withOpacity(0.12),
                          shape: BoxShape.circle,
                        ),
                        child: Icon(widget.icon, size: 14, color: widget.color),
                      ),
                      const SizedBox(width: 8),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            widget.label,
                            style: GoogleFonts.plusJakartaSans(
                              color: AppColors.textPrimary,
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          Text(
                            widget.subtitle,
                            style: GoogleFonts.inter(
                              color: AppColors.textMuted,
                              fontSize: 10,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    transitionBuilder: (child, anim) => ScaleTransition(scale: anim, child: child),
                    child: Text(
                      '${widget.count}',
                      key: ValueKey<int>(widget.count),
                      style: GoogleFonts.plusJakartaSans(
                        color: widget.count > 0 ? Colors.white : AppColors.textMuted,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              // Ratio progress bar
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Ratio',
                        style: GoogleFonts.inter(
                          color: AppColors.textMuted,
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Text(
                        _percentageStr,
                        style: GoogleFonts.inter(
                          color: widget.color,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: Stack(
                      children: [
                        Container(
                          height: 5,
                          width: double.infinity,
                          color: AppColors.cardElevated,
                        ),
                        AnimatedFractionallySizedBox(
                          duration: const Duration(milliseconds: 400),
                          curve: Curves.easeOutCubic,
                          widthFactor: _ratio.clamp(0.0, 1.0),
                          child: Container(
                            height: 5,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  widget.color.withOpacity(0.7),
                                  widget.color,
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
