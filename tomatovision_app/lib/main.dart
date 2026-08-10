import 'package:flutter/material.dart';
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
        scaffoldBackgroundColor: const Color(0xFF0F172A),
      ),
      home: const ViewfinderScreen(),
    );
  }
}