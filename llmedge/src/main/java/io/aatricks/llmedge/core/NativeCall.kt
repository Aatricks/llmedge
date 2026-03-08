package io.aatricks.llmedge.core

internal object NativeCall {
    inline fun <T> binding(libraryName: String, detail: String, block: () -> T): T =
        try {
            block()
        } catch (e: UnsatisfiedLinkError) {
            throw NativeBindingException(libraryName, detail, e)
        }

    fun requireHandle(handle: Long, modelPath: String, detail: String): Long {
        if (handle == 0L) {
            throw ModelLoadException(modelPath, detail)
        }
        return handle
    }

    inline fun <T> optional(defaultValue: T, block: () -> T): T =
        try {
            block()
        } catch (_: UnsatisfiedLinkError) {
            defaultValue
        }
}