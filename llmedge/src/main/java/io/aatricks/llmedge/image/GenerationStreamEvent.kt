package io.aatricks.llmedge.image

import android.graphics.Bitmap
import io.aatricks.llmedge.core.ProgressEvent

sealed interface GenerationStreamEvent {
    data class Progress(val update: ProgressEvent.Step) : GenerationStreamEvent
    data class Completed(val frames: List<Bitmap>) : GenerationStreamEvent
}
