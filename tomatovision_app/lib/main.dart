import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'core/constants/app_colors.dart';
import 'presentation/screens/viewfinder_screen.dart';
import 'services/tflite_helper.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await getTfliteHelper().loadModel();
  runApp(const TomatoVisionApp());
}

class TomatoVisionApp extends StatelessWidget {
  const TomatoVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TomatoVision AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: AppColors.background,
        colorScheme: const ColorScheme.dark(
          primary: AppColors.brandPrimary,
          secondary: AppColors.tierFresh,
          surface: AppColors.surface,
        ),
        textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme),
        appBarTheme: const AppBarTheme(
          backgroundColor: AppColors.background,
          elevation: 0,
        ),
      ),
      home: const ViewfinderScreen(),
    );
  }
}