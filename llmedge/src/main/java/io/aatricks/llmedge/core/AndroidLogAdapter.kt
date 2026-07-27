package io.aatricks.llmedge.core

internal object AndroidLogAdapter {
    private const val DEBUG_LEVEL = "D"
    private const val INFO_LEVEL = "I"
    private const val WARN_LEVEL = "W"
    private const val ERROR_LEVEL = "E"

    /** Receives every line alongside logcat, so diagnostics can outlive the process. */
    internal fun interface Sink {
        fun onLog(level: String, tag: String, message: String, throwable: Throwable?)
    }

    @Volatile private var sink: Sink? = null

    internal fun setSink(sink: Sink?) {
        this.sink = sink
    }

    private fun mirror(level: String, tag: String, message: String, throwable: Throwable? = null) {
        val target = sink ?: return
        try {
            target.onLog(level, tag, message, throwable)
        } catch (_: Throwable) {
            // A failing diagnostic sink must never break the call it is observing.
        }
    }

    private val logClass: Class<*>? by lazy {
        try {
            Class.forName("android.util.Log")
        } catch (_: Throwable) {
            null
        }
    }

    private val debugMethod by lazy { logClass?.getMethod("d", String::class.java, String::class.java) }
    private val infoMethod by lazy { logClass?.getMethod("i", String::class.java, String::class.java) }
    private val warnMethod by lazy { logClass?.getMethod("w", String::class.java, String::class.java) }
    private val errorMethod by lazy {
        logClass?.getMethod(
            "e",
            String::class.java,
            String::class.java,
            Throwable::class.java,
        )
    }

    fun d(tag: String, message: String) = log(debugMethod, DEBUG_LEVEL, tag, message)

    fun i(tag: String, message: String) = log(infoMethod, INFO_LEVEL, tag, message)

    fun w(tag: String, message: String) = log(warnMethod, WARN_LEVEL, tag, message)

    fun e(tag: String, message: String, throwable: Throwable? = null) {
        mirror(ERROR_LEVEL, tag, message, throwable)
        val method = errorMethod
        if (method != null) {
            try {
                method.invoke(null, tag, message, throwable)
                return
            } catch (_: Throwable) {
                // Fall back to stderr below.
            }
        }

        System.err.println("$ERROR_LEVEL/$tag: $message")
        throwable?.let { System.err.println(it.stackTraceToString()) }
    }

    private fun log(method: java.lang.reflect.Method?, level: String, tag: String, message: String) {
        mirror(level, tag, message)
        if (method != null) {
            try {
                method.invoke(null, tag, message)
                return
            } catch (_: Throwable) {
                // Fall back to stdout below.
            }
        }

        println("$level/$tag: $message")
    }
}