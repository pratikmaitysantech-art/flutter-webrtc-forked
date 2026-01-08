import 'package:flutter/services.dart';

/// NEW ADDITION: Helper class to control face detection overlay on video stream
/// This allows you to enable/disable the bounding box overlay that is sent
/// to the remote peer through WebRTC.
class FaceDetectionOverlayControl {
  static const MethodChannel _channel =
      MethodChannel('FlutterWebRTC.Method');

  /// Enable or disable face detection overlay on the outgoing video stream
  /// When enabled, the remote peer will see the green bounding boxes on your video
  /// When disabled, only face detection data is sent to Flutter (for local UI)
  /// but the actual video stream is unmodified
  static Future<void> setOverlayEnabled(bool enabled) async {
    try {
      await _channel.invokeMethod('setFaceDetectionOverlay', {
        'enabled': enabled,
      });
    } catch (e) {
      print('Error setting face detection overlay: $e');
    }
  }
}

