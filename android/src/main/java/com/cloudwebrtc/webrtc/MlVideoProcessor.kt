package com.cloudwebrtc.webrtc

/* CHANGES LOG:
 * 1. Added: MlVideoProcessor for background face detection on camera video frames.
 * 2. Added: Hooks for optional mask rendering using Canvas over detected faces.
 */

// NEW ADDITION START
// ADDED: Kotlin-based video frame processor that uses ML Kit Face Detection on a background thread.

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.os.Handler
import android.os.HandlerThread
import android.util.Log

import com.cloudwebrtc.webrtc.video.LocalVideoTrack
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions

import org.webrtc.VideoFrame
import org.webrtc.YuvHelper

import java.nio.ByteBuffer
import java.util.Arrays

/**
 * NEW ADDITION:
 * Lightweight face-detection processor that plugs into the existing LocalVideoTrack
 * external processing pipeline. It runs ML Kit face detection on a separate thread so
 * it does not block the WebRTC capture thread.
 *
 * NOTE: This processor currently does NOT mutate the underlying video frame; it is
 * intended for tracking / callbacks and can be extended to apply visual effects.
 */
class MlVideoProcessor(
    context: Context,
    private val listener: FaceDetectionListener? = null
) : LocalVideoTrack.ExternalVideoFrameProcessing {

    interface FaceDetectionListener {
        fun onFacesDetected(faces: List<Face>, frame: VideoFrame)
        fun onNoFacesDetected(frame: VideoFrame)
    }

    private val applicationContext: Context = context.applicationContext

    // Background thread for ML Kit so we don't block WebRTC's capture thread.
    private val thread = HandlerThread("FlutterWebRTC-ML-FaceDetection")
    private val handler: Handler

    // ML Kit face detector instance tuned for real-time performance.
    private val faceDetector: FaceDetector

    init {
        thread.start()
        handler = Handler(thread.looper)

        val realTimeOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()

        faceDetector = FaceDetection.getClient(realTimeOpts)
    }

    /**
     * Called on the WebRTC capture thread for every frame.
     * We offload the heavy work to a background Handler thread and immediately
     * return the original frame so the media pipeline is not delayed.
     */
    override fun onFrame(frame: VideoFrame): VideoFrame {
        val buffer = frame.buffer
        val i420 = buffer.toI420()

        // MODIFIED: Added null-safety check for I420 buffer conversion.
        if (i420 == null) {
            Log.w(TAG, "Failed to convert video frame buffer to I420 format, skipping face detection")
            return frame
        }

        // Copy width/height before posting to another thread.
        val width = i420.width
        val height = i420.height
        val rotation = frame.rotation

        val inputImage = try {
            inputImageFromI420(i420, rotation)
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to convert I420 frame to InputImage", e)
            i420.release()
            return frame
        }

        // We can now safely release the I420 buffer as ML Kit works on its own copy.
        i420.release()

        handler.post {
            processFrame(inputImage, frame, width, height)
        }

        // IMPORTANT: Return the original frame to keep the pipeline intact.
        return frame
    }

    private fun processFrame(
        inputImage: InputImage,
        frame: VideoFrame,
        width: Int,
        height: Int
    ) {
        faceDetector
            .process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    Log.d(TAG, "Face detection success: ${faces.size} face(s)")
                    listener?.onFacesDetected(faces, frame)

                    // This is the hook where mask rendering can be implemented
                    // using OpenGL or Canvas with the detection results.
                    //
                    // For example (Canvas-based pseudocode):
                    //   1. Convert the YUV frame to a Bitmap.
                    //   2. Draw masks aligned to each Face.boundingBox on a Canvas.
                    //   3. Convert the Bitmap back to a VideoFrame (I420) if you want to
                    //      feed the modified frame into the WebRTC pipeline.
                    //
                    // The actual visual rendering strategy depends on your app's needs
                    // and can be implemented on top of this callback.
                    renderMasksIfNeeded(faces, width, height)
                } else {
                    Log.d(TAG, "No faces detected in current frame")
                    listener?.onNoFacesDetected(frame)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection failed", e)
            }
    }

    /**
     * Placeholder hook for Canvas / OpenGL mask rendering.
     * Implementers can override this class or wrap it to draw masks onto a
     * dedicated surface, texture, or overlay view.
     */
    protected open fun renderMasksIfNeeded(faces: List<Face>, width: Int, height: Int) {
        // NEW ADDITION:
        // This method is intentionally left as a no-op in the base implementation.
        // Apps can subclass MlVideoProcessor and override this to render masks
        // using Canvas or OpenGL with the provided face bounding boxes.
        if (faces.isNotEmpty()) {
            val boxes = faces.joinToString(prefix = "[", postfix = "]") { face: Face ->
                val b: Rect = face.boundingBox
                "(${b.left},${b.top})-(${b.right},${b.bottom})"
            }
            Log.d(TAG, "Mask rendering hook invoked for faces: $boxes on $width x $height")
        }
    }

    /**
     * Convert an I420 buffer into an InputImage in NV21 format for ML Kit.
     */
    private fun inputImageFromI420(i420Buffer: VideoFrame.I420Buffer, rotation: Int): InputImage {
        val y: ByteBuffer = i420Buffer.dataY
        val u: ByteBuffer = i420Buffer.dataU
        val v: ByteBuffer = i420Buffer.dataV
        val width = i420Buffer.width
        val height = i420Buffer.height

        val strides = intArrayOf(
            i420Buffer.strideY,
            i420Buffer.strideU,
            i420Buffer.strideV
        )

        val chromaWidth = (width + 1) / 2
        val chromaHeight = (height + 1) / 2
        val minSize = width * height + chromaWidth * chromaHeight * 2

        // Allocate a heap buffer so we can get the underlying byte[]
        val yuvBuffer = ByteBuffer.allocate(minSize)

        // NOTE: NV21 is like NV12 but with U and V swapped, so we use I420ToNV12
        // helper and swap U/V buffers accordingly.
        YuvHelper.I420ToNV12(
            y, strides[0],
            v, strides[2],
            u, strides[1],
            yuvBuffer,
            width,
            height
        )

        // Remove any leading zeros to avoid shifted images.
        val cleanedArray = Arrays.copyOfRange(yuvBuffer.array(), 0, minSize)

        return InputImage.fromByteArray(
            cleanedArray,
            width,
            height,
            rotation,
            ImageFormat.NV21
        )
    }

    /**
     * Should be called when camera capture is stopped to cleanly shut down
     * the background thread and ML Kit detector.
     */
    fun dispose() {
        try {
            faceDetector.close()
        } catch (e: Throwable) {
            Log.e(TAG, "Error while closing face detector", e)
        }

        try {
            thread.quitSafely()
        } catch (e: Throwable) {
            Log.e(TAG, "Error while stopping ML thread", e)
        }
    }

    companion object {
        private const val TAG = "MlVideoProcessor"
    }
}
