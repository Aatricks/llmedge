package io.aatricks.llmedge

import java.io.File
import java.util.concurrent.ConcurrentHashMap
import org.junit.Assume

internal object DesktopNativeTestSupport {
    private val loadedLibraries = ConcurrentHashMap.newKeySet<String>()

    fun resolveLibraryPath(
        envName: String,
        defaultRelativePath: String,
    ): String =
        System.getenv(envName)
            ?: System.getProperty(envName)
            ?: File(System.getProperty("user.dir") ?: ".", defaultRelativePath).absolutePath

    fun requireEnabled() {
        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue(
            "Native loading is disabled. Provide the host native library path and leave llmedge.disableNativeLoad unset.",
            disableNativeLoad != "true",
        )
    }

    fun requireAndLoadLibrary(libPath: String) {
        val libFile = File(libPath)
        Assume.assumeTrue("Native library not found at $libPath", libFile.exists())
        if (loadedLibraries.add(libFile.absolutePath)) {
            System.load(libFile.absolutePath)
        }
    }

    fun requireEnabledAndLoadLibrary(
        envName: String,
        defaultRelativePath: String,
    ): String {
        requireEnabled()
        val libPath = resolveLibraryPath(envName, defaultRelativePath)
        requireAndLoadLibrary(libPath)
        return libPath
    }
}
