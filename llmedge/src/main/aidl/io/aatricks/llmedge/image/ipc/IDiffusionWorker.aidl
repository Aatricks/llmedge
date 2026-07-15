package io.aatricks.llmedge.image.ipc;

import android.os.Bundle;
import io.aatricks.llmedge.image.ipc.WorkerInitConfig;
import io.aatricks.llmedge.image.ipc.IpcImageRequest;
import io.aatricks.llmedge.image.ipc.IpcVideoRequest;
import io.aatricks.llmedge.image.ipc.IDiffusionResultCallback;
import io.aatricks.llmedge.image.ipc.IDiffusionVideoCallback;
import io.aatricks.llmedge.image.ipc.IpcUpscaleRequest;

interface IDiffusionWorker {
    int getPid();
    // Idempotent; re-sent after every (re)connect. Rebuilds the worker engine when config changes.
    void initialize(in WorkerInitConfig config);
    oneway void generateImage(in IpcImageRequest request, IDiffusionResultCallback callback);
    oneway void generateVideo(in IpcVideoRequest request, IDiffusionVideoCallback callback);
    oneway void cancelGeneration();
    // Debug builds only (FLAG_DEBUGGABLE); throws SecurityException otherwise.
    void installFaultInjection(in Bundle args);
    oneway void upscaleImage(in IpcUpscaleRequest request, IDiffusionResultCallback callback);
}
