package io.aatricks.llmedge.image.ipc;

import io.aatricks.llmedge.image.ipc.PhaseUpdate;
import io.aatricks.llmedge.image.ipc.IpcVideoResult;
import io.aatricks.llmedge.image.ipc.IpcFailure;

oneway interface IDiffusionVideoCallback {
    void onPhase(in PhaseUpdate update);
    void onProgress(String message, int current, int total);
    void onCompleted(in IpcVideoResult result);
    void onFailed(in IpcFailure failure);
}
