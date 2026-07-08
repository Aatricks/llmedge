package io.aatricks.llmedge.image.ipc;

import io.aatricks.llmedge.image.ipc.PhaseUpdate;
import io.aatricks.llmedge.image.ipc.IpcImageResult;
import io.aatricks.llmedge.image.ipc.IpcFailure;

oneway interface IDiffusionResultCallback {
    void onPhase(in PhaseUpdate update);
    void onCompleted(in IpcImageResult result);
    void onFailed(in IpcFailure failure);
}
