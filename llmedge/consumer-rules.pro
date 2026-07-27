# Consumer ProGuard rules for llmedge library
# These rules will be included in apps that use this library

# Keep all vision and OCR API classes
-keep class io.aatricks.llmedge.vision.** { *; }
-keep interface io.aatricks.llmedge.vision.** { *; }


# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# The native layer resolves these by name via FindClass/GetMethodID, so R8 must
# not rename or remove them. The rule above only covers classes that themselves
# declare native methods, which none of these do.
-keep class io.aatricks.llmedge.speech.stt.Whisper$TranscriptionSegment {
    <init>(int, long, long, java.lang.String);
}
-keep interface io.aatricks.llmedge.speech.stt.Whisper$ProgressCallback { *; }
-keep interface io.aatricks.llmedge.speech.stt.Whisper$SegmentCallback { *; }
-keep interface io.aatricks.llmedge.image.diffusion.VideoProgressCallback { *; }
-keepclassmembers class * implements io.aatricks.llmedge.speech.stt.Whisper$ProgressCallback {
    void onProgress(int);
}
-keepclassmembers class * implements io.aatricks.llmedge.speech.stt.Whisper$SegmentCallback {
    void onNewSegment(int, long, long, java.lang.String);
}
-keepclassmembers class * implements io.aatricks.llmedge.image.diffusion.VideoProgressCallback {
    void onProgress(int, int, int, int, float);
}
# BarkTTS wraps its callback in an anonymous object whose onProgress is only
# ever invoked from native, so R8 would otherwise strip the method body.
-keepclassmembers class io.aatricks.llmedge.speech.tts.** {
    void onProgress(int, int);
}

# Keep ML Kit classes
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_text_recognition.** { *; }
-keep class com.google.android.gms.internal.mlkit_text_recognition_common.** { *; }

# Suppress warnings for optional dependencies
-dontwarn com.google.mlkit.**

# Keep enums
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep Parcelable
-keepclassmembers class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator CREATOR;
}

# Keep serializable classes
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}

# Keep data classes
-keep class io.aatricks.llmedge.vision.ImageSource$* { *; }
-keep class io.aatricks.llmedge.vision.OcrParams { *; }
-keep class io.aatricks.llmedge.vision.OcrResult { *; }
-keep class io.aatricks.llmedge.vision.VisionParams { *; }
-keep class io.aatricks.llmedge.vision.VisionResult { *; }
-keep class io.aatricks.llmedge.vision.ImageUnderstandingResult { *; }
-keep enum io.aatricks.llmedge.vision.VisionMode { *; }