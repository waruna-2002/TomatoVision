import 'package:flutter_test/flutter_test.dart';
import 'package:tomatovision_app/main.dart';

void main() {
  testWidgets('TomatoVisionApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const TomatoVisionApp());
    expect(find.text('TomatoVision AI'), findsOneWidget);
  });
}

